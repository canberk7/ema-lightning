"""Playhead: one background loop that owns the GPU and runs every caller's work in shared batches.

Every text is cut into sentences. Each sentence is planned (the text stage, which fixes its exact length),
thought (the aligner and the four sampling steps, all at once) and decoded (windows of four seconds of audio;
a stream's very first window is one second, so its first audio comes fast). There are two queues, both
strictly first come, first served:

    queue 1, before the model:    sentences waiting to be thought
    queue 2, before the decoder:  windows waiting to be decoded

Each turn the loop plans everything that arrived during the last turn, thinks up to one batch of sentences
from the front of queue 1, decodes up to one batch of windows from the front of queue 2, and hands out the
audio in order, each sentence followed by its pause. Whatever does not fit in a batch stays at the front
for the next turn. On a GPU a decode batch holds windows of one decode size: it stops at the first window
of another size, which goes first in the next turn, so a stream's one-second window is never stretched to
four seconds and nothing is reordered. A CPU has no decode sizes, so there a batch holds either windows of
one second or less, or longer ones, and stops at the first window of the other kind. The loop sleeps when
there is nothing to do.

A caller who hangs up leaves both queues. If a stage fails, only the callers in that batch get the error.
"""
import collections
import queue
import threading

from .engine import FIRST_WINDOW, RATE, WINDOW, batches, windows

DONE = object()


class Request:
    """One caller: their sentences, and an outbox that yields audio tensors, then DONE or an exception."""

    def __init__(self, pieces, speed, first):
        self.pieces, self.speed, self.first = pieces, speed, first
        self.outbox = queue.SimpleQueue()
        self.closed = False
        self.left = 0  # windows still to deliver

    def cancel(self):
        self.closed = True

    def finish(self, error=None):
        if not self.closed:
            self.closed = True
            self.outbox.put(DONE if error is None else error)

    def send(self, piece, last, audio):
        self.outbox.put(audio)
        if last:
            if piece.pause:
                self.outbox.put(audio.new_zeros(round(piece.pause * RATE)))
            piece.h = piece.dur = piece.latents = None
        self.left -= 1
        if self.left == 0:
            self.finish()


class Playhead:
    def __init__(self, engine, batch_size):
        self.engine, self.batch_size = engine, batch_size
        self.cond = threading.Condition()
        self.inbox = []
        self.sentences = collections.deque()  # queue 1: (request, piece)
        self.windows = collections.deque()  # queue 2: (request, piece, span, last window of its piece)
        self.thread = None

    def submit(self, pieces, speed, first=WINDOW):
        """Queue one caller's sentences; returns the request whose outbox carries their audio.

        `first` is the length in frames of the first window of the first sentence: four seconds for say(),
        one second for stream(). Every other window is four seconds.
        """
        request = Request(pieces, speed, first)
        if not pieces:
            request.finish()
            return request
        with self.cond:
            self.inbox.append(request)
            if self.thread is None:
                self.thread = threading.Thread(target=self._loop, name="ema-playhead", daemon=True)
                self.thread.start()
            self.cond.notify()
        return request

    def _loop(self):
        while True:
            with self.cond:
                while not self.inbox and not self.sentences and not self.windows:
                    self.cond.wait()
                new, self.inbox = self.inbox, []
            try:
                with self.engine.lock:
                    self._turn(new)
            except Exception as error:  # anything outside the stages: fail every caller in flight, keep serving
                callers = {id(r): r for r in new}
                callers.update({id(r): r for r, *_ in [*self.sentences, *self.windows]})
                self._fail(callers.values(), error)

    def _turn(self, new):
        size = max(1, int(self.batch_size()))
        self._plan([r for r in new if not r.closed], size)
        self._drop_closed()
        self._think(size)
        self._decode(size)

    def _plan(self, requests, size):
        """Plan every newcomer's sentences, then put them at the back of queue 1 in arrival order."""
        by_speed = collections.defaultdict(list)
        for r in requests:
            by_speed[r.speed].append(r)
        for speed, group in by_speed.items():
            owner = {id(p): r for r in group for p in r.pieces}
            pieces = sorted((p for r in group for p in r.pieces), key=lambda p: p.letters)
            for batch in batches(pieces, size):
                try:
                    self.engine.plan(batch, speed)
                except Exception as error:
                    self._fail({id(owner[id(p)]): owner[id(p)] for p in batch}.values(), error)
        for r in requests:
            if not r.closed:
                for i, p in enumerate(r.pieces):
                    p.spans = windows(p.frames, r.first if i == 0 else WINDOW)
                r.left = sum(len(p.spans) for p in r.pieces)
                self.sentences.extend((r, p) for p in r.pieces)

    def _think(self, size):
        """Think the batch at the front of queue 1; their windows go to the back of queue 2."""
        batch = [self.sentences.popleft() for _ in range(min(size, len(self.sentences)))]
        if not batch:
            return
        try:
            self.engine.think([p for _, p in batch])
        except Exception as error:
            self._fail({id(r): r for r, _ in batch}.values(), error)
            return
        for r, p in batch:
            if not r.closed:
                self.windows.extend((r, p, span, i == len(p.spans) - 1) for i, span in enumerate(p.spans))

    def _decode(self, size):
        """Decode the batch at the front of queue 2 and hand each window to its caller."""
        sized = bool(getattr(self.engine, "decode_sizes", ()))
        batch, kind = [], None
        while self.windows and len(batch) < size:
            _, p, span, _ = self.windows[0]
            this = self.engine.decode_size(p, span) if sized else span[1] - span[0] <= FIRST_WINDOW
            if kind is not None and this != kind:
                break
            kind = this
            batch.append(self.windows.popleft())
        if not batch:
            return
        try:
            audio = self.engine.decode([(p, span) for _, p, span, _ in batch])
        except Exception as error:
            self._fail({id(r): r for r, *_ in batch}.values(), error)
            return
        for (r, p, _, last), a in zip(batch, audio, strict=True):
            if not r.closed:
                r.send(p, last, a)

    def _fail(self, requests, error):
        for r in requests:
            r.finish(error)
        self._drop_closed()

    def _drop_closed(self):
        self.sentences = collections.deque(x for x in self.sentences if not x[0].closed)
        self.windows = collections.deque(x for x in self.windows if not x[0].closed)
