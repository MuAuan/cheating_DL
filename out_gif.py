from PIL import Image, ImageDraw

s=30
#N=20
#RATE = 11025
#fr = RATE
#fn=51200*N/50  #*RATE/44100
#fs=fn/fr
#list=[0,0.2,0.5,1,2,5,10,20,50]
#list=[2,3,4,5,6,7,8,9,10]
images = []
for i in range(1,30,1):
    im = Image.open('output/category5/image'+str(i)+'.jpg') 
    im =im.resize(size=(800, 800), resample=Image.LANCZOS)  #- NEAREST - BOX - BILINEAR - HAMMING - BICUBIC - LANCZOS
    images.append(im)
    
images[0].save('./output/category5-30.gif', save_all=True, append_images=images[1:s], duration=500*1.0, loop=0)    
