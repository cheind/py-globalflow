```
ffmpeg -r 20 -i C:\dev\py-globalflow\tmp\ts18_keypoints_reid\img_%06d.png  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -r 20 -pix_fmt yuv420p out.mp4
```