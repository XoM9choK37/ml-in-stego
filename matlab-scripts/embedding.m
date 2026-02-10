clc; clear all;

payload = single(0.4);

fprintf('Embedding using Matlab file\n');

for i = 1:10000
    cover_number = i;
    cover_name = sprintf('%d.pgm', cover_number);

    cover_path = ['../../BOSSbase_1.01_256x256/', cover_name];
    % original images were resized to 256x256 via "new_img = imresize(img, [256, 256])" (with default parameters in imresize)

    stego_image = S_UNIWARD(cover_path, payload);
    
    stego_path = ['../stego_images/', cover_name];
    imwrite(uint8(stego_image), stego_path)
end

fprintf('Embedding has been completed\n');
