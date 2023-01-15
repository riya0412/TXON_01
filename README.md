# TXON_01
TXON Machine Learning internship tast 01 - Black and White Image colourization

Image Colourization is the process of taking an input black and white image and then producing an image output colourized image.

I have used OpenCV package for this project.
The technique I have covered here is from Zhang et al.’s 2016 ECCV paper, Colorful Image Colorization.Zhang et al. decided to attack the problem of image colorization by using Convolutional Neural Networks to “hallucinate” what an input grayscale image would look like when colorized.

I have used LAB colour space;
       The L channel encodes lightness intensity only
       The a channel encodes green-red.
       And the b channel encodes blue-yellow
			 
