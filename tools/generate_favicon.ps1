<#
.SYNOPSIS
    Render the browser-tab favicon from the Savanna Institute circle logo.

.DESCRIPTION
    Crops the source logo's transparent margins (measured per side, so the mark ends up centred
    without being stretched), resizes the square that remains to 512x512, and writes it to the
    frontend's public assets. A 32x32 copy is written beside it so tab-size legibility can be
    checked by eye.

    The output bitmap is constructed as 32bppArgb and cleared to Transparent before drawing:
    drawing onto a default-format bitmap loses the source alpha and fills the margin black.

.EXAMPLE
    powershell -File tools/generate_favicon.ps1 -Source "$HOME/Downloads/FullColorLogo_circleonly.png"
#>
param(
    [string]$Source = "$HOME\Downloads\FullColorLogo_circleonly.png",
    [string]$OutDir = (Join-Path $PSScriptRoot "..\packages\tcip-web\frontend\public\assets"),
    [int]$CropTop = 77,
    [int]$CropBottom = 99,
    [int]$CropLeft = 142,
    [int]$CropRight = 142,
    [int]$Size = 512,
    [int]$PreviewSize = 32
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Drawing

if (-not (Test-Path -LiteralPath $Source)) {
    throw "source image not found: $Source"
}
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path

$src = [System.Drawing.Image]::FromFile((Resolve-Path -LiteralPath $Source).Path)
try {
    $cropX = $CropLeft
    $cropY = $CropTop
    $cropW = $src.Width - $CropLeft - $CropRight
    $cropH = $src.Height - $CropTop - $CropBottom
    if ($cropW -le 0 -or $cropH -le 0) {
        throw "crop margins exceed the source dimensions ($($src.Width)x$($src.Height))"
    }
    # Keep the result square without scaling either axis on its own: trim the longer side evenly.
    $side = [Math]::Min($cropW, $cropH)
    $cropX += [int](($cropW - $side) / 2)
    $cropY += [int](($cropH - $side) / 2)
    $srcRect = New-Object System.Drawing.Rectangle $cropX, $cropY, $side, $side

    function Save-Square {
        param([System.Drawing.Image]$Image, [System.Drawing.Rectangle]$Rect, [int]$Edge, [string]$Path)

        $bmp = New-Object System.Drawing.Bitmap $Edge, $Edge, ([System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
        try {
            $g = [System.Drawing.Graphics]::FromImage($bmp)
            try {
                $g.Clear([System.Drawing.Color]::Transparent)
                $g.CompositingMode = [System.Drawing.Drawing2D.CompositingMode]::SourceOver
                $g.CompositingQuality = [System.Drawing.Drawing2D.CompositingQuality]::HighQuality
                $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
                $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
                $g.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
                $dest = New-Object System.Drawing.Rectangle 0, 0, $Edge, $Edge
                $g.DrawImage($Image, $dest, $Rect, [System.Drawing.GraphicsUnit]::Pixel)
            }
            finally { $g.Dispose() }
            $bmp.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
        }
        finally { $bmp.Dispose() }
        Write-Host "wrote $Path"
    }

    Save-Square -Image $src -Rect $srcRect -Edge $Size -Path (Join-Path $OutDir "si_logo_favicon.png")
    Save-Square -Image $src -Rect $srcRect -Edge $PreviewSize -Path (Join-Path $OutDir "si_logo_favicon_preview32.png")
}
finally { $src.Dispose() }
