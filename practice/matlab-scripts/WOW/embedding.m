clc; clear all;

payload = 0.4;

params.p = -1;

fprintf('Embedding using Matlab file\n');

for i = 1:10000
    cover_number = i;
    cover_name = sprintf('%d.pgm', cover_number);

    cover_path = fullfile('..', '..', 'BOSSbase_1.01_256x256', cover_name)
    % original images were resized to 256x256 via "new_img = imresize(img, [256, 256])" (with default parameters in imresize)

    [stego_image, distortion] = WOW(imread(cover_path), payload, params);
	
    stego_path = fullfile('..', 'stego_images', cover_name);
    imwrite(uint8(stego_image), stego_path)
end

fprintf('Embedding has been completed\n');
