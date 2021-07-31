Generate videos from `track_poses.py`

```
ffmpeg -r 20 -i indir\img_%06d.png  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -r 20 -pix_fmt yuv420p out.mp4
```