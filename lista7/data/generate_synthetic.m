%load example data (half sphere)
load data_sphere;

% some calibration matrices
calib_synthetic = zeros(3, 3, 2);
calib_synthetic(:,:,1) = [6 0 7; 0 8 9; 0 0 1];
calib_synthetic(:,:,2) = [5 0 3; 0 5 2; 0 0 1];

% some rotation and translation
R1 = eye(3); t1 = zeros(3, 1);
R2 = getRotationMatrix([degtorad(10), degtorad(20), degtorad(30)]); t2 = [0.1 0.2 0.3]';

% project to image 1
points2d_synthetic(:,:,1) = calib_synthetic(:,:,1) * [R1 t1] * points3d_synthetic;
points2d_synthetic(:,:,1) = points2d_synthetic(:,:,1) ./ repmat(points2d_synthetic(3,:,1), 3, 1);

% project to image 2
points2d_synthetic(:,:,2) = calib_synthetic(:,:,2) * [R2 t2] * points3d_synthetic;
points2d_synthetic(:,:,2) = points2d_synthetic(:,:,2) ./ repmat(points2d_synthetic(3,:,2), 3, 1);

points2d = points2d_synthetic;
K1 = calib_synthetic(:,:,1);
K2 = calib_synthetic(:,:,2);
% clean up 
clear R1 R2 t1 t2;