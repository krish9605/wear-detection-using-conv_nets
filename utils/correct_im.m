clc
clear
folds = {'14'}

for folder_dir = folds
    folder_dir = folder_dir{1}
    file_extension = '*.jpg';
    
    full_name = fullfile(folder_dir,file_extension);
    img_files = dir(full_name);
    nImgs = length(img_files);
    
    start_img = 1;
    
    nImg = length(img_files);
    
    img_indices =  start_img:nImg;
    
    %% [IMAGE, CONTOUR, HEIGHT, WEIGHT]
    for i = img_indices
        im_loc = fullfile(folder_dir,img_files(i).name);
        img = imread(im_loc);
        k = size(img);
        if(k(1)<k(2))
            img = permute(img,[2 1 3]);
            imwrite(img,[im_loc(1:(end-4)),'_imcor.png'])
        else
            imwrite(img,[im_loc(1:(end-4)),'_imcor.png'])
        end
    end
end
