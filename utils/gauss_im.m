clc
clear

folder_dir = 'data/0';
file_extension = '*.png';

full_name = fullfile(folder_dir,file_extension);
img_files = dir(full_name);

for i = 1:length(img_files)
    
    im_loc = fullfile(folder_dir,img_files(i).name);
    if(is_mask(img_files(i).name))
        img = imread(im_loc);
        img = imgaussfilt(img,30);
        img_files(i).name(end-4)='g';
        imwrite(img,[folder_dir,'/',img_files(i).name])
    end
end
disp('ran?')
function bool_val = is_mask(img_str)
    bool_val = img_str(end-6:end) == 'bin.png';
end