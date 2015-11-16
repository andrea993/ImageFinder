% File  with the image to load (256 colors image)
clear; clc;
ImgFile='bmp256col.JPG'; 
fmt='JPEG'; % file format (optional)

info=imfinfo(ImgFile,fmt) % get file information

%Read the file
imRGBuint = imread(ImgFile,fmt); 

%Make the pictures collection
hf=figure(1); clf; colormap('default');
nr=5;nc=3;

%How to show a colored image
% The image(I) function needs that the I elements are in [0,1] if
% double, in [0,255] if uint8 and in [0,65535] if uint16.

% In this example we will use double images, so in [0,1] 
% The RGB images are rappresented by a 3D matrix that
% on the 3rd dimensione has the 2D matrices: R, G, B .
% The white and black images are composed by 3 EQUALS matrices R,G,B

% Image of int in [0,255]
ha=subplot(nr,nc,1); image(imRGBuint); title('original uint RGB')

% Image of double in [0-1]
imRGB=double(imRGBuint)/255;
ha=subplot(nr,nc,2); image(imRGB); title('original double RGB')


% View a coloured deatil
im_detail=imRGB(62:78,20:40,:);
im_detail=imRGB(66:73,43:64,:);
ha=subplot(nr,nc,3); image(im_detail); title('detail')

% Decomposition of the RGB picture in its 3 base colors
R=imRGB(:,:,1); 
G=imRGB(:,:,2);
B=imRGB(:,:,3);

% Recomposition of the 3 components in monochrome 3D bitmap
RGB2Image=@(R,G,B) cat(3,R,G,B);

imR=RGB2Image(R*1,G*0,B*0);
ha=subplot(nr,nc,4);image(imR); title('R')
imG=RGB2Image(R*0,G*1,B*0);
ha=subplot(nr,nc,5);image(imG); title('G')
imB=RGB2Image(R*0,G*0,B*1);
ha=subplot(nr,nc,6);image(imB); title('B')


% Make a B&W 2D matrix (!you can't show it with image(I) function!)
RGB2Matrix=@(R,G,B) (R+G+B)/3;
Image2Matrix=@(Image) (Image(:,:,1)+Image(:,:,2)+Image(:,:,3))/3;

Matrix2BW=@(mBW) cat(3,mBW,mBW,mBW);

mBW=RGB2Matrix(R,G,B); % It is a matrix, not an image
imBW=Matrix2BW(mBW); % Return a BW image where R=G=B
ha=subplot(nr,nc,7);image(imBW); title('B&W')


%Filtering of the image y=(x-xmin)/(xmax-xmin) to saturate
%from 0 to 1. This filter increases the contrast 
contrast=@(x,xmin,xmax) max(0,min(1,(x-xmin)/(xmax-xmin)));

imBW1=contrast(imBW,0.1,0.9);
ha=subplot(nr,nc,8);image(imBW1); title('B&W enhanced')
imRGB1=contrast(imRGB,0.1,0.9);
ha=subplot(nr,nc,9);image(imRGB1); title('RGB enhanced')


% DFTtrasformation of an image
% If the FFT2 is called on an RGB image, it is made on each of the 3
% components
x=mBW;
X=fft2(x); % Scale factor (row x column)
Xdisp=log(X); s=max(abs(Xdisp(:))); Xdisp=abs(Xdisp)/s; % Preparing to show
ha=subplot(nr,nc,10);image(Matrix2BW(Xdisp)); title('FFT2')

% Antitransformed
x=min(1,max(0,ifft2(X)));
ha=subplot(nr,nc,11);image(Matrix2BW(x)); title('iFFT2')

% Convolution
h=[-1,-2,-1;0,0,0;1,2,1]/8; % Sobel filter
%xf=filter2(h,x,'same')+filter2(h',x,'same'); % NB: filter rovescia specularmente il filtro
xf=conv2(x,h,'same')+conv2(x,h','same');

% Gain normalization
Normal=@(x) (x-min(x(:)))./(max(x(:)-min(x(:))));
xf=Normal(xf);
ha=subplot(nr,nc,12);image(Matrix2BW(xf)); title('Sobel filter')

% Convolution with fft
HV=fft2(h,size(x,1),size(x,2)); % Transformed reported to the same size of x
HO=fft2(h',size(x,1),size(x,2)); % Transformed reported to the same size of x
Z=(HV+HO).*X;
z=ifft2(Z);
z=Normal(z);
ha=subplot(nr,nc,13);image(Matrix2BW(z)); title('Sobel filter fft')

% Convolution with a detail
h=Image2Matrix(im_detail); % 
xf=filter2(h-mean(h(:)),x-mean(x(:)),'same');
xf=Normal(xf);
ha=subplot(nr,nc,14);image(Matrix2BW(xf)); title('Detail filter')
[i,j]=find(xf==max(xf(:)));
hold;
plot(j,i,'or');

% Convolution with fft of a detail
X=fft2(mBW-mean(mBW(:)));
H=fft2(h-mean(h(:)),size(x,1),size(x,2)); % Transformed reported to the same size of x
Z=(conj(H)).*X;
z=ifft2(Z);
z=Normal(z);
ha=subplot(nr,nc,15);image(Matrix2BW(z)); title('Detail filter fft')
[i,j]=find(z==max(z(:)));
hold;
plot(j,i,'or');

