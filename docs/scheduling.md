# Many callers

One `EMA` serves every caller in the process. Call `say()` and `stream()` from as
many threads as you like, for example one per request or phone call; there is no
server or database to run. Behind them, one background loop called Playhead owns
the GPU and keeps two queues, both strictly first come, first served:

- **Queue 1, before the model:** sentences waiting to be thought. A caller's text
  is cut into sentences, and all of them join the back of the queue in order.
- **Queue 2, before the decoder:** windows waiting to become audio. When a sentence
  has been thought, its windows join the back of this queue.

Each turn, Playhead:

1. Plans everything that arrived during the last turn, so every sentence's
   length is known.
2. Thinks up to one batch of sentences from the front of queue 1.
3. Decodes up to one batch of windows from the front of queue 2.
4. Hands each caller their audio, in order, with each sentence's pause after it.

The batch size is `best_batch_size()`. A batch is shared by however many callers
are at the front of the queue, and whatever does not fit stays at the front for
the next turn. New work keeps queueing while the GPU runs; Playhead never waits to
fill a batch, and it sleeps when both queues are empty.

## Windows

A sentence is decoded in windows of four seconds, each with eight frames of the
neighbouring audio on either side, so the joins are seamless. A stream's first
sentence starts with a one-second window, so its first audio comes early. `say()`
returns once all of a text's audio is back; `stream()` hands each window over as
soon as it is decoded.

On a GPU, every window is padded to one of two decoder sizes, 48 or 120 frames,
and a decode batch holds windows of one size: it stops at the first window of
another size, which goes first in the next turn. Nothing is reordered. This keeps
a stream's one-second window from being stretched to four seconds, and keeps the
number of shapes small for the graphs of `lightning()`.

## Batches on the fast path

`lightning()` records graphs at batch sizes 1, 2, 4 and every multiple of 8 up to
the batch size, and each batch is padded only to the nearest of them. A lone
stream's first window runs as a batch of one; a full queue runs as a full batch.

## Long texts, hang-ups and failures

A long text queues all of its sentences, so callers behind it wait for them: cut
very long inputs before sending them if that matters to you.

When a stream's iterator is closed or let go (for example, by breaking out of the
`for` loop), its entries leave both queues.

If a batch fails (for example, out of memory), only the callers in that batch get
the error; Playhead keeps serving everyone else.

## The same audio

Because sentences and windows from different callers share GPU passes, the last
digits of the audio can differ from a run alone. On an RTX PRO 6000 the difference
measured about 60 dB quieter than the speech, which is inaudible; the tests require
at least 40 dB.

To measure call times with many callers on your GPU, run `benches/callers.py`.
