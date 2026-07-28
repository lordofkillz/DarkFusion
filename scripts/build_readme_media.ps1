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
color=c=0x090c10:s=1280x912:d=2.5,setsar=1,fps=24[v0base];
[v0base]drawtext=fontfile='$boldFont':text='LABEL FASTER. TRAIN CLEANER.':fontcolor=white:fontsize=58:x=(w-text_w)/2:y=345,drawtext=fontfile='$font':text='DarkFusion 5.2  |  One YOLO workflow':fontcolor=0x56e0c2:fontsize=31:x=(w-text_w)/2:y=440,fade=t=in:st=0:d=0.3,fade=t=out:st=2.2:d=0.3[v0];
[0:v]scale=1280:912,setsar=1,fps=24,drawbox=x=0:y=820:w=iw:h=92:color=black@0.74:t=fill,drawtext=fontfile='$boldFont':text='BOUNDING BOXES':fontcolor=white:fontsize=38:x=42:y=842,fade=t=in:st=0:d=0.18,fade=t=out:st=1.17:d=0.18[v1];
[1:v]scale=1280:912,setsar=1,fps=24,drawbox=x=0:y=820:w=iw:h=92:color=black@0.74:t=fill,drawtext=fontfile='$boldFont':text='POLYGON SEGMENTATION':fontcolor=white:fontsize=38:x=42:y=842,fade=t=in:st=0:d=0.18,fade=t=out:st=1.17:d=0.18[v2];
[2:v]scale=1280:912,setsar=1,fps=24,drawbox=x=0:y=820:w=iw:h=92:color=black@0.74:t=fill,drawtext=fontfile='$boldFont':text='ORIENTED BOXES':fontcolor=white:fontsize=38:x=42:y=842,fade=t=in:st=0:d=0.18,fade=t=out:st=1.17:d=0.18[v3];
[3:v]scale=1280:912,setsar=1,fps=24,drawbox=x=0:y=820:w=iw:h=92:color=black@0.74:t=fill,drawtext=fontfile='$boldFont':text='POSE + KEYPOINTS':fontcolor=white:fontsize=38:x=42:y=842,fade=t=in:st=0:d=0.18,fade=t=out:st=1.17:d=0.18[v4];
[4:v]scale=1280:912,setsar=1,fps=24,drawbox=x=0:y=820:w=iw:h=92:color=black@0.76:t=fill,drawtext=fontfile='$boldFont':text='LEFT-CLICK A PREVIEW  >  FLASH THE EXACT ANNOTATION':fontcolor=white:fontsize=30:x=(w-text_w)/2:y=846,drawbox=x=858:y=355:w=106:h=107:color=white@0.95:t=4:enable='between(t,0.65,1.15)',drawbox=x=858:y=455:w=106:h=107:color=white@0.95:t=4:enable='between(t,2.35,2.85)',drawbox=x=858:y=555:w=106:h=107:color=white@0.95:t=4:enable='between(t,4.05,4.55)',fade=t=in:st=0:d=0.25,fade=t=out:st=5.75:d=0.25[v5];
[5:v]scale=1280:912,setsar=1,fps=24,drawbox=x=0:y=820:w=iw:h=92:color=black@0.74:t=fill,drawtext=fontfile='$boldFont':text='TRAIN  |  TUNE  |  EXPORT':fontcolor=white:fontsize=36:x=42:y=839,drawtext=fontfile='$font':text='PyTorch  |  ONNX  |  TensorRT':fontcolor=0x56e0c2:fontsize=24:x=740:y=850,fade=t=in:st=0:d=0.25,fade=t=out:st=2.75:d=0.25[v6];
color=c=0x090c10:s=1280x912:d=2.5,setsar=1,fps=24[v7base];
[v7base]drawtext=fontfile='$boldFont':text='FROM RAW FRAMES TO CLEAN WEIGHTS':fontcolor=white:fontsize=50:x=(w-text_w)/2:y=360,drawtext=fontfile='$font':text='DarkFusion 5.2':fontcolor=0x56e0c2:fontsize=34:x=(w-text_w)/2:y=450,fade=t=in:st=0:d=0.3,fade=t=out:st=2.2:d=0.3[v7];
[v0][v1][v2][v3][v4][v5][v6][v7]concat=n=8:v=1:a=0,format=yuv420p[outv]
"@

& $Ffmpeg -y `
    -loop 1 -t 1.35 -i (Join-Path $samples "boxes.png") `
    -loop 1 -t 1.35 -i (Join-Path $samples "Segmentation.png") `
    -loop 1 -t 1.35 -i (Join-Path $samples "OBB.png") `
    -loop 1 -t 1.35 -i (Join-Path $samples "pose.png") `
    -i (Join-Path $samples "review_preview_interaction.mp4") `
    -loop 1 -t 3 -i (Join-Path $samples "TrainExport.png") `
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
