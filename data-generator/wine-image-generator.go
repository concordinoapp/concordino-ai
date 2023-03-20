package main  
 
import "fmt" 
import _ "image/png"

func main() {  
	// create image’s background
	bgImg := image.NewRGBA(image.Rect(0, 0, bgProperty.Width, bgProperty.Length))
	// set the background color
	draw.Draw(bgImg, bgImg.Bounds(), &image.Uniform{bgProperty.BgColor}, image.ZP, draw.Src)
}
