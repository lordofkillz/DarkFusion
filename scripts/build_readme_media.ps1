param(
    [string]$Ffmpeg = "ffmpeg"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$samples = Join-Path $repoRoot "samples"
$font = "C\:/Windows/Fonts/segoeui.ttf"
$boldFont = "C\:/Windows/Fonts/segoeuib.ttf"
$video = Join-Path $samples "darkfusion_walkthrough.mp4"
$gif = Join-Path $samples "darkfusion2.gif"

$filter = @"
color=c=0x090c10:s=1280x912:d=2[v0base];
[v0base]drawtext=fontfile='$boldFont':text='UltraDarkFusion 5.2':fontcolor=white:fontsize=62:x=(w-text_w)/2:y=350,drawtext=fontfile='$font':text='Build, label, review, train, and export YOLO datasets':fontcolor=0x56e0c2:fontsize=30:x=(w-text_w)/2:y=440,fade=t=in:st=0:d=0.3,fade=t=out:st=1.7:d=0.3[v0];
[0:v]scale=1280:912,setsar=1,drawbox=x=0:y=820:w=iw:h=92:color=black@0.72:t=fill,drawtext=fontfile='$boldFont':text='1  Bounding boxes':fontcolor=white:fontsize=38:x=42:y=842,fade=t=in:st=0:d=0.25,fade=t=out:st=2.75:d=0.25[v1];
[1:v]scale=1280:912,setsar=1,drawbox=x=0:y=820:w=iw:h=92:color=black@0.72:t=fill,drawtext=fontfile='$boldFont':text='2  Polygon segmentation':fontcolor=white:fontsize=38:x=42:y=842,fade=t=in:st=0:d=0.25,fade=t=out:st=2.75:d=0.25[v2];
[2:v]scale=1280:912,setsar=1,drawbox=x=0:y=820:w=iw:h=92:color=black@0.72:t=fill,drawtext=fontfile='$boldFont':text='3  Oriented bounding boxes':fontcolor=white:fontsize=38:x=42:y=842,fade=t=in:st=0:d=0.25,fade=t=out:st=2.75:d=0.25[v3];
[3:v]scale=1280:912,setsar=1,drawbox=x=0:y=820:w=iw:h=92:color=black@0.72:t=fill,drawtext=fontfile='$boldFont':text='4  Pose and keypoints':fontcolor=white:fontsize=38:x=42:y=842,fade=t=in:st=0:d=0.25,fade=t=out:st=2.75:d=0.25[v4];
[4:v]scale=1280:912,setsar=1,drawbox=x=0:y=820:w=iw:h=92:color=black@0.72:t=fill,drawtext=fontfile='$boldFont':text='5  Review labels and training issues':fontcolor=white:fontsize=38:x=42:y=842,fade=t=in:st=0:d=0.25,fade=t=out:st=2.75:d=0.25[v5];
color=c=0x090c10:s=1280x912:d=2.5[v6base];
[v6base]drawtext=fontfile='$boldFont':text='One workflow. Every YOLO task.':fontcolor=white:fontsize=54:x=(w-text_w)/2:y=360,drawtext=fontfile='$font':text='DarkFusion':fontcolor=0x56e0c2:fontsize=34:x=(w-text_w)/2:y=445,fade=t=in:st=0:d=0.3,fade=t=out:st=2.2:d=0.3[v6];
[v0][v1][v2][v3][v4][v5][v6]concat=n=7:v=1:a=0,format=yuv420p[outv]
"@

& $Ffmpeg -y `
    -loop 1 -t 3 -i (Join-Path $samples "boxes.png") `
    -loop 1 -t 3 -i (Join-Path $samples "Segmentation.png") `
    -loop 1 -t 3 -i (Join-Path $samples "OBB.png") `
    -loop 1 -t 3 -i (Join-Path $samples "pose.png") `
    -loop 1 -t 3 -i (Join-Path $samples "Review.png") `
    -filter_complex $filter -map "[outv]" -r 24 -c:v libx264 -preset medium -crf 22 `
    -movflags +faststart $video
if ($LASTEXITCODE -ne 0) {
    throw "MP4 generation failed with exit code $LASTEXITCODE."
}

& $Ffmpeg -y -i $video `
    -vf "fps=6,scale=960:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4" `
    -loop 0 $gif
if ($LASTEXITCODE -ne 0) {
    throw "GIF generation failed with exit code $LASTEXITCODE."
}

Write-Host "Created:"
Write-Host "  $video"
Write-Host "  $gif"
