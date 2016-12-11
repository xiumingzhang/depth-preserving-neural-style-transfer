img1 = rgb2gray(imread('results/chicago_input.jpg'));
img2 = rgb2gray(imread('results/chicago_starry_feifei.jpg'));
img3 = rgb2gray(imread('results/chicago_starry_ours.jpg'));

[mssim ssim_map] = ssim(img1, img2);
disp(mssim);
imshow(max(0, ssim_map).^4)

[mssim ssim_map] = ssim(img1, img3)
disp(mssim);
imshow(max(0, ssim_map).^4)
