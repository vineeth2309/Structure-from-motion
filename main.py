import numpy as np
import math
import glob
import cv2
import matplotlib.pyplot as plt
from mayavi import mlab
import open3d as o3d
from scipy.optimize import least_squares
import pytransform3d.transformations as pt
import pytransform3d.camera as pc
import pytransform3d.visualizer as pv

class sfm:
    def __init__(self, images_path='data/fountain-P11/images/'):
        self.image_paths = glob.glob(images_path+"*.jpg")[:3]
        print(self.image_paths)
        self.K = np.array([[2759.48, 0, 1520.69], [0, 2764.16, 1006.81], [0, 0 , 1]])
        self.surf = cv2.xfeatures2d.SURF_create()
        self.sift = cv2.xfeatures2d.SIFT_create()
        index_params = dict(algorithm = 1, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        self.R_t_0 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
        self.R_t_1 = np.empty((3,4))
        self.obs_mtx = np.eye(4)
        self.P1 = np.matmul(self.K, self.R_t_0)
        self.P2 = np.empty((3,4))
        self.pts = None
        self.start = False
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.run()

    def read_images(self):
        self.images = []
        for img in self.image_paths:
            self.images.append(cv2.imread(img))

    def feature_match(self, img1, img2):
        kp1, desc1 = self.sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
        kp2, desc2 = self.sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
        matches = self.matcher.knnMatch(desc1,desc2, k=2)
        kps1, kps2, colors = [], [], []
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                kps1.append(kp1[m.queryIdx].pt)
                kps2.append(kp2[m.trainIdx].pt)
                color_coords = np.array(kp1[m.queryIdx].pt)
                color_idx = [int(color_coords[0]), int(color_coords[1])]
                color_idx = [math.floor(color_coords[1]), math.floor(color_coords[0])]
                colors.append(img1[color_idx[0], color_idx[1], :])
        kps1, kps2, colors = np.array(kps1), np.array(kps2), np.flip(np.array(colors))
        return kps1, kps2, colors

    def create_transformation_mtx(self, R, t):
        return np.hstack((R,t))

    def homogenous_4x4coords(self, mtx):
        return np.vstack((mtx, np.array([0,0,0,1])))

    def homogenous_3x1coords(self, mtx):
        return np.hstack((mtx, np.ones((mtx.shape[0], 1))))

    def update_obs_mtx(self, R, t):
        T = self.homogenous_4x4coords(self.create_transformation_mtx(R, t))
        self.obs_mtx = np.matmul(np.linalg.inv(T), self.obs_mtx)

    def computepose(self, kpts1, kpts2, colors):
        F, mask = cv2.findFundamentalMat(kpts1, kpts2, cv2.FM_RANSAC)
        kpts1 = kpts1[mask.ravel()==1]
        kpts2 = kpts2[mask.ravel()==1]
        colors = colors[mask.ravel()==1]
        E = np.matmul(np.matmul(np.transpose(self.K), F), self.K)
        retval, R, t, mask = cv2.recoverPose(E, kpts2, kpts1, self.K)
        self.R_t_1 = self.create_transformation_mtx(R, t)   # Transformation from frame 2 to frame 1
        self.P2 = np.matmul(self.K, self.R_t_1)
        return kpts1, kpts2, colors, F, R, t

    def triangulate_pts(self, kpts1, kpts2, R, t, colors):
        pt_3d = cv2.triangulatePoints(self.P1, self.P2, kpts1.T, kpts2.T)
        # T = self.homogenous_4x4coords(self.create_transformation_mtx(R, t))
        pt_3d = np.matmul(self.obs_mtx, pt_3d)
        pt_3d /= pt_3d[3]
        if self.start == False:
            self.pts = pt_3d.T[:, :3]
            self.colors = np.zeros((kpts2.shape[0], 3))
            self.start = True
        else:
            self.pts = np.concatenate((self.pts, pt_3d.T[:, :3]))
            self.colors = np.concatenate((self.colors, colors))

    def reprojection_loss_function(self, opt_variables, points_2d, num_pts):
        '''
        opt_variables --->  Camera Projection matrix + All 3D points
        '''
        P = opt_variables[0:12].reshape(3,4)
        point_3d = opt_variables[12:].reshape((num_pts, 4))
        rep_error = []
        for idx, pt_3d in enumerate(point_3d):
            pt_2d = np.array([points_2d[0][idx], points_2d[1][idx]])
            reprojected_pt = np.matmul(P, pt_3d)
            reprojected_pt /= reprojected_pt[2]
            rep_error.append(pt_2d - reprojected_pt[0:2])
        return np.array(rep_error).ravel()

    def rep_error_fn(self, opt_variables, points_2d, num_pts):
        P = opt_variables[0:12].reshape(3,4)
        point_3d = opt_variables[12:].reshape((num_pts, 4))
        rep_error = []
        for idx, pt_3d in enumerate(point_3d):
            pt_2d = np.array([points_2d[0][idx], points_2d[1][idx]])
            reprojected_pt = np.matmul(P, pt_3d)
            reprojected_pt /= reprojected_pt[2]
            rep_error.append(pt_2d - reprojected_pt[0:2])
        print("MEAN ERROR: ",np.sum(rep_error, axis=0))

    def bundle_adjustment(self, points_3d, points_2d, projection_matrix):
        opt_variables = np.hstack((projection_matrix.ravel(), points_3d.ravel(order="F")))
        num_points = len(points_2d[0])
        corrected_values = least_squares(self.reprojection_loss_function, opt_variables, args=(points_2d,num_points))
        print("The optimized values \n" + str(corrected_values))
        P = corrected_values.x[0:12].reshape(3,4)
        points_3d = corrected_values.x[12:].reshape((num_points, 4))
        return P, points_3d

    def draw_epilines(self, F, kpts1, kpts2, img1, img2):
        lines1 = cv2.computeCorrespondEpilines(kpts2.reshape(-1,1,2), 2, F)
        lines1 = lines1.reshape(-1,3)
        r,c,_ = img1.shape
        for r, pt1, pt2 in zip(lines1[:50], kpts1[:50], kpts2[:50]):
            color = tuple(np.random.randint(0,255,3).tolist())
            x0,y0 = map(int, [0, -r[2]/r[1] ])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            img1 = cv2.line(img1, (x0,y0), (x1,y1), color, 3)
            img1 = cv2.circle(img1,tuple([int(pt1[0]), int(pt1[1])]),5,color,-1)
            img2 = cv2.circle(img2,tuple([int(pt2[0]), int(pt2[1])]),5,color,-1)

        lines2 = cv2.computeCorrespondEpilines(kpts1.reshape(-1,1,2), 2, F)
        lines2 = lines2.reshape(-1,3)
        r,c,_ = img2.shape
        for r, pt1, pt2 in zip(lines2[:50], kpts1[:50], kpts2[:50]):
            color = tuple(np.random.randint(0,255,3).tolist())
            x0,y0 = map(int, [0, -r[2]/r[1] ])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            img2 = cv2.line(img2, (x0,y0), (x1,y1), color, 3)
            img2 = cv2.circle(img2,tuple([int(pt1[0]), int(pt1[1])]),5,color,-1)
            img1 = cv2.circle(img1,tuple([int(pt2[0]), int(pt2[1])]),5,color,-1)
        return img1,img2

    def show_matches(self, img1, img2, img1pts, img2pts):
        concat_img = np.hstack((img1, img2))
        for pt1, pt2 in zip(img1pts[:50], img2pts[:50]):
            cv2.circle(concat_img, (int(pt1[0]), int(pt1[1])), 7, (0,255,0), -1)
            cv2.circle(concat_img, (int(pt2[0])+img1.shape[1], int(pt2[1])), 7, (0,255,0), -1)
            cv2.line(concat_img, (int(pt1[0]), int(pt1[1])),
                                 (int(pt2[0])+img1.shape[1], int(pt2[1])),
                                 color=(255, 0, 0), thickness=2)
        cv2.imshow("FEATURE MATCHES", cv2.resize(concat_img, (1280,720)))
        cv2.waitKey(1)
        pass

    def warp_affine(self, img1, img2, pts1, pts2, F, R, t):
        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape
        # R, _ = cv2.Rodrigues(R)
        _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), F, imgSize=(w1, h1))
        H1, H2, P1, P2, Q, _, _ = cv2.stereoRectify(cameraMatrix1=self.K, distCoeffs1=np.array([]), cameraMatrix2=self.K, distCoeffs2=np.array([]), imageSize=(w1, h1), R=self.P2, T=t)
        Q = np.float32([[1,0,0,-w1/2.0],
                        [0,-1,0,h1/2.0],
                        [0,0,0,-self.K[0][0]],
                        [0,0,1,0]])
        Q = np.float32([[1,0,0,0],
                        [0,-1,0,0],
                        [0,0,-self.K[0][0],0], #Focal length multiplication obtained experimentally.
                        [0,0,0,1]])
        imgL = cv2.cvtColor(cv2.warpPerspective(img1, H1, (w1, h1)), cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(cv2.warpPerspective(img2, H2, (w2, h2)), cv2.COLOR_BGR2GRAY)
        # stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        win_size = 5
        min_disp = -1
        max_disp = 63 #min_disp * 9
        num_disp = max_disp - min_disp # Needs to be divisible by 16
        stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
         numDisparities = num_disp,
         blockSize = 5,
         uniquenessRatio = 5,
         speckleWindowSize = 5,
         speckleRange = 5,
         disp12MaxDiff = 1,
         P1 = 8*3*win_size**2,#8*3*win_size**2,
         P2 =32*3*win_size**2) #32*3*win_size**2)
        disparity =  stereo.compute(imgL,imgR)

        points_3D = cv2.reprojectImageTo3D(disparity, Q)
        colors = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        mask_map = disparity > disparity.min()
        output_points = points_3D[mask_map]
        output_colors = colors[mask_map]
        cv2.imshow("RECT", cv2.resize(np.hstack((imgL, imgR)),(1280,720)))
        plt.imshow(disparity)
        self.viz_3d(output_points, output_colors)
        plt.show()
        cv2.waitKey()

    def draw_coord_frame(self):
        points = [[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
        points = np.matmul(self.obs_mtx, np.array(points).T).T[:, :3]
        lines = [[0, 1], [0, 2], [0, 3]]
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        self.vis.add_geometry(line_set)

    def draw_pose(self):
        cam_points = [[0, 0, 0, 1], [-0.5, -0.5, 1, 1], [-0.5, 0.5, 1, 1], [0.5, 0.5, 1, 1], [0.5, -0.5, 1, 1]]
        cam_points = np.matmul(self.obs_mtx, np.array(cam_points).T).T[:, :3]
        cam_lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [1, 4]]
        cam_colors = [[1, 0, 0] for i in range(len(cam_lines))]
        cam_colors = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(cam_points)
        line_set.lines = o3d.utility.Vector2iVector(cam_lines)
        line_set.colors = o3d.utility.Vector3dVector(cam_colors)
        self.vis.add_geometry(line_set)

    def viz_3d(self, pt_3d, colors):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pt_3d)
        pcd.colors = o3d.utility.Vector3dVector(colors/255)
        self.vis.add_geometry(pcd)
        self.vis.run()

    def run(self):
        self.read_images()
        for i in range(len(self.images) - 1):
            img1, img2 = self.images[i], self.images[i + 1]
            img1pts, img2pts, colors = self.feature_match(img1, img2)
            img1pts, img2pts, colors, F, R, t = self.computepose(img1pts, img2pts, colors)
            self.draw_pose()
            self.triangulate_pts(img1pts, img2pts, R, t, colors)
            self.show_matches(img1, img2, img1pts, img2pts)
            self.update_obs_mtx(R, t)
            # print(self.obs_mtx)
            # self.obs_mtx = np.matmul(self.homogenous_4x4coords(self.R_t_1), self.obs_mtx)
            # img1, img2 = self.draw_epilines(F, img1pts, img2pts, img1, img2)
            # concat_img = np.hstack((img1, img2))
            # cv2.imshow("IMAGES", cv2.resize(concat_img, (1280, 720)))
            # cv2.waitKey()

        self.viz_3d(self.pts, self.colors)


if __name__ == "__main__":
    sfm()
