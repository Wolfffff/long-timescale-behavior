# Long Timescale Drosophila Behavior - Imaging

## Imaging

### Motif

Loopbio's [Motif](http://loopbio.com/recording/)

See the basic config for our setup: [recnode.yml.backup](recnode.yml.backup).

For the most recent config, see [recnode.yml](recnode.yml) and the camera specific configs which are named recnode.SERIAL.yml.

We're currently recording to Loopbio's [imgstore](https://github.com/loopbio/imgstore). For ease of use with SLEAP, we're concatenating files using ffmpeg. See [here.](https://trac.ffmpeg.org/wiki/Concatenate)

### labcams

On easy and cheap way to record with realtime GPU compression is to use labcams from Joao Couto. This is significantly more CPU intensive than Motif but also free and open source.

## SLEAP
