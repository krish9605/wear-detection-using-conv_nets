clc
clear

folder_dir = '2';
file_extension = '*.jpg';

full_name = fullfile(folder_dir,file_extension);
img_files = dir(full_name);
nImgs = length(img_files);s

rects = zeros(nImgs,4);
conts = zeros(10000,2*length(img_files));
fig = figure;
start_img = 1;
end_img = 3;

nImg = length(img_files);
contour_counter = 1;

img_indices =  start_img:nImg;
binSize = MakeBinaryMask(folder_dir,img_files);

binaryMasks = zeros([binSize,length(img_indices)]);
%% [IMAGE, CONTOUR, HEIGHT, WEIGHT]
for i = img_indices
    im_loc = fullfile(folder_dir,img_files(i).name);
    img = imread(im_loc);
    k = size(img);
    if(k(1)<k(2))
        img = permute(img,[2 1 3]);

    end    
    ax = gca;
    imshow(img,'Parent',ax);
    hold(ax,'on')
    start_val = 1;
    while(1)
        % if first time 

        if(start_val == 1)
            exit_val = 0;
            binaryImage = zeros();

        else
            title('waiting')
            k = waitforbuttonpress;
            exit_val = double(get(gcf,'CurrentCharacter'))-28;
            title('done')
        end
        
        if(exit_val == 0)
            hFH = imfreehand('Closed',false);
            xy=hFH.getPosition();
            plot(xy(:,1),xy(:,2),'r*','LineWidth',3)
            nBox = size(xy,1);
            
            conts(1,2*contour_counter-1) = i;
            conts(1,2*contour_counter) = nBox;
            conts(2:nBox+1,[2*contour_counter-1,2*contour_counter]) = xy;

            % Get mask
            binaryImage = binaryImage + hFH.createMask();

            start_val = 0;
            contour_counter = contour_counter+1;
        else
            contour_counter = 1;
%             save([im_loc(1:(end-4)),'.mat'],'binaryImage')
            imwrite(binaryImage,[im_loc(1:(end-4)),'_bin.png'])
           % binaryMasks(:,:,i) = binaryImage;
            break;
        end

    end
    
    pause(0.3)
    hold(ax,'off')
    
end
close all
save(['all_rects_',num2str(start_img),'_',num2str(end_img)],'conts','img_files')
function val_returned = myfun(src,event)
if(strcmp(event.Key,'return'))
    val_returned = 1;
end
end


function size_img = MakeBinaryMask(folder_dir,img_files)
    im_loc = fullfile(folder_dir,img_files(1).name);
    img = imread(im_loc);
    size_img = size(img);
    size_img = [size_img(1),size_img(2)];
end
%     rect = getrect;
%     rectangle('Position',rect, 'EdgeColor','r','LineWidth',3)
%     rects(i,:) = [rect([1,2]),rect([3,4])+  rect([1,2])];