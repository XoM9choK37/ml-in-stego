clc; clear all;

cover_number = 1;
cover_name = sprintf('%d.pgm', cover_number);

cover_path = fullfile('..', '..', 'BOSSbase_1.01_256x256', cover_name);
stego_path = fullfile('..', 'stego_images', cover_name);

stego_image = imread(stego_path);
cover_image = imread(cover_path);

figure;
subplot(1, 3, 1); imshow(cover_image); title('cover');
subplot(1, 3, 2); imshow(uint8(stego_image)); title('stego');
subplot(1, 3, 3); imshow((double(stego_image) - double(cover_image) + 1) / 2); title('Embedding changes: +1 = white, -1 = black');
fprintf('Change rate: %.4f\n', sum(cover_image(:) ~= stego_image(:)) / numel(cover_image));
