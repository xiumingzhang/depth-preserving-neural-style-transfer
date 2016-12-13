imList = {'1R1A2120', 'chicago', 'IMG_0155', 'IMG_0165', 'IMG_0176', 'IMG_0193', 'IMG_20160221_135719_1', 'IMG_20160221_135735_2', 'IMG_20160221_142327', 'IMG_20160305_164711', 'IMG_20160306_124732', 'IMG_20160306_131711_1'}; % , 'IMG_20160307_132402'};
styleList = {'starry', 'muse', 'wave', 'composition'};

mssimListOurs = [];
mssimListFeifei = [];
for i=1:length(imList)

    inputName = sprintf('results/%s_input.jpg', imList{i});
    
    for j=1:length(styleList)
        feifeiName = sprintf('results/%s_%s_feifei.jpg', imList{i}, styleList{j});
        ourName = sprintf('results/%s_%s_ours.jpg', imList{i}, styleList{j});
    

img1 = rgb2gray(imread(inputName));
img2 = rgb2gray(imread(feifeiName));
img3 = rgb2gray(imread(ourName));
img1 = imresize(img1, size(img2));
[mssim ~] = ssim(img1, img2);
mssimListFeifei = [mssimListFeifei, mssim];
%imshow(max(0, ssim_map).^4);

[mssim ~] = ssim(img1, img3);
mssimListOurs = [mssimListOurs, mssim];
%imshow(max(0, ssim_map).^4);
end
end
fh = figure(1);
set(fh, 'Position', [100, 100, 500, 895]);
y = [mssimListOurs; mssimListFeifei]';
bar(y);
legend('Ours', 'Johnson et al.')