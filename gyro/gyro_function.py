import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

def get_static(height = 1080, width = 1920, ratio = 0.1):
    static_options = {}
    static_options["active_array_width"] = 4032
    static_options["active_array_height"] = 3024
    static_options["crop_window_width"] = 4032
    static_options["crop_window_height"] = 2272
    static_options["num_grid_rows"] = 12
    static_options["num_grid_cols"] = 12
    static_options["dim_homography"] = 9
    static_options["width"] = width  # frame width.
    static_options["height"] = height # frame height
    # static_options["fov"] = 1.27 # sensor_width/sensor_focal_length
    static_options["cropping_ratio"] = 0.0 #ratio # normalized cropping ratio at each side. 
    return static_options

# Quaternion: [x, y, z, w]

def norm_quat(quat):
    norm_quat = LA.norm(quat)   
    if norm_quat > 1e-6:
        quat = quat / norm_quat   
        #     [0 norm_quat norm_quat - 1e-6]
    else:
        # print('bad len for Reciprocal')
        quat = np.array([0,0,0,1])
    return quat

def torch_norm_quat(quat, USE_CUDA = True):
    # Method 1:
    batch_size = quat.size()[0]
    quat_out = Variable(torch.zeros((batch_size, 4), requires_grad=True))
    if USE_CUDA == True:
        quat_out = quat_out.cuda()
    for i in range(batch_size):
        norm_quat = torch.norm(quat[i])   
        if norm_quat > 1e-6:        
            quat_out[i] = quat[i] / norm_quat  
            #     [0 norm_quat norm_quat - 1e-6]
        else:
            quat_out[i,:3] = quat[i,:3] * 0
            quat_out[i,3] = quat[i,3] / quat[i,3]

    # Method 2:
    # quat = quat / (torch.unsqueeze(torch.norm(quat, dim = 1), 1) + 1e-6) # check norm
    return quat_out

# 60 quat = (cos(half_angle), u(sin(half_angle))), where u is axis

def ConvertAxisAngleToQuaternion(axis, angle):
    if LA.norm(axis) > 1e-6 and angle > 1e-6: 
        axis = axis/LA.norm(axis)  
    half_angle = angle*0.5  
    sin_half_angle = np.sin(half_angle)
    quat = np.array([sin_half_angle* axis[0], sin_half_angle* axis[1], sin_half_angle* axis[2], np.cos(half_angle)]) #https://blog.csdn.net/qq_40851561/article/details/120478280?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~aggregatepage~first_rank_ecpm_v1~rank_v31_ecpm-1-120478280.pc_agg_new_rank&utm_term=sin+%E5%90%91%E9%87%8F%E7%9B%B8%E4%B9%98&spm=1000.2123.3001.4430

    return norm_quat(quat)

def ConvertAxisAngleToQuaternion_no_angle(axis):
    angle = LA.norm(axis)  
    if LA.norm(axis) > 1e-6: 
        axis = axis/LA.norm(axis)  
    half_angle = angle*0.5  
    sin_half_angle = np.sin(half_angle)
    quat = np.array([sin_half_angle* axis[0], sin_half_angle* axis[1], sin_half_angle* axis[2], np.cos(half_angle)])

    return norm_quat(quat)

def torch_ConvertAxisAngleToQuaternion(axis, USE_CUDA = True):
    batch_size = axis.size()[0]

    angle = torch.norm(axis[:,:3], dim = 1)

    half_angle = angle * 0.5 
    sin_half_angle = torch.sin(half_angle)
    quats = Variable(torch.zeros((batch_size, 4), requires_grad=True))
    norm_axis = axis[:,:3] * 1
    if USE_CUDA:
        quats = quats.cuda()
    for i in range(batch_size):
        if angle[i] > 1e-6:
            norm_axis[i] = axis[i,:3]/angle[i]
    quats[:, :3] = sin_half_angle * norm_axis
    quats[:, 3] = torch.cos(half_angle)
    return torch_norm_quat(quats)

def ConvertQuaternionToAxisAngle(quat):
    quat = quat/LA.norm(quat)   
    axis_norm = LA.norm(quat[0:3])
    axis = np.array([0.0, 0.0, 0.0])
    if axis_norm < 1e-6:
        angle = 0   
    else:
        axis_norm_reciprocal = 1/axis_norm   
        axis[0] = quat[0] * axis_norm_reciprocal   
        axis[1] = quat[1] * axis_norm_reciprocal   
        axis[2] = quat[2] * axis_norm_reciprocal   
        angle = 2 * np.arccos(quat[3])
    return [axis, angle]

def ConvertQuaternionToAxisAngle_no_angle(quat):
    quat = quat/LA.norm(quat)   
    axis_norm = LA.norm(quat[0:3])
    axis = np.array([0.0, 0.0, 0.0])  
    if axis_norm > 1e-6:
        axis_norm_reciprocal = 1 / axis_norm * 2 *  np.arccos(quat[3])
        axis[0] = quat[0] * axis_norm_reciprocal   
        axis[1] = quat[1] * axis_norm_reciprocal   
        axis[2] = quat[2] * axis_norm_reciprocal   
    return axis

def torch_ConvertQuaternionToAxisAngle(quat, USE_CUDA = True):
    batch_size = quat.size()[0]
    axis_angle = Variable(torch.zeros((batch_size, 4), requires_grad=True))
    if USE_CUDA:
        axis_angle = axis_angle.cuda()
    for i in range(batch_size): 
        axis_norm = torch.norm(quat[i, 0:3])
        if axis_norm > 1e-6:
            axis_norm_reciprocal = 1/axis_norm  * 2 * torch.acos(quat[i,3])
            axis_angle[i,0] = quat[i,0] * axis_norm_reciprocal   
            axis_angle[i,1] = quat[i,1] * axis_norm_reciprocal   
            axis_angle[i,2] = quat[i,2] * axis_norm_reciprocal   
    return axis_angle

def train_ConvertQuaternionToAxisAngle(quat):
    out = np.zeros(4)
    out[:3] = ConvertQuaternionToAxisAngle_no_angle(quat)
    return out

def AngularVelocityToQuat(angular_v, dt):
    length = LA.norm(angular_v)  
    if length < 1e-6:
        angular_v = np.array([1, 0, 0])  
        print('bad length')
    else:
        angular_v = angular_v/length  
    quat = ConvertAxisAngleToQuaternion(angular_v, length*dt) 
    return quat

def QuaternionProduct(q1, q2):
    x1 = q1[0]  
    y1 = q1[1]   
    z1 = q1[2]   
    w1 = q1[3]   

    x2 = q2[0]  
    y2 = q2[1]  
    z2 = q2[2]  
    w2 = q2[3]  

    quat = np.zeros(4)
    #https://blog.csdn.net/qq_40851561/article/details/120478280?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~aggregatepage~first_rank_ecpm_v1~rank_v31_ecpm-1-120478280.pc_agg_new_rank&utm_term=sin+%E5%90%91%E9%87%8F%E7%9B%B8%E4%B9%98&spm=1000.2123.3001.4430
    #  (3-16) jv zhen cheng fa

    quat[3] =  w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2  
    quat[0] =  w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2  
    quat[1] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2  
    quat[2] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2 

    return norm_quat(quat)

def torch_QuaternionProduct(q1, q2, USE_CUDA = True):
    x1 = q1[:,0]  
    y1 = q1[:,1]   
    z1 = q1[:,2]   
    w1 = q1[:,3]   

    x2 = q2[:,0]  
    y2 = q2[:,1]  
    z2 = q2[:,2]  
    w2 = q2[:,3]  

    batch_size = q1.size()[0]
    quat = Variable(torch.zeros((batch_size, 4), requires_grad=True))
    if USE_CUDA == True:
        quat = quat.cuda()
    
    quat[:,3] =  w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2  
    quat[:,0] =  w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2  
    quat[:,1] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2  
    quat[:,2] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2  

    quat = torch_norm_quat(quat)

    return quat

def ProcessGyroRotation(gyro_data):
    num_inputs = np.shape(gyro_data)[0]
    quats = np.zeros((num_inputs, 4))  
    quats[0,:] = np.array([0, 0, 0, 1])
    for i in range(1, num_inputs):
        dt = (gyro_data[i, 0] - gyro_data[i-1, 0])*1e-9  
        quat = AngularVelocityToQuat(gyro_data[i, 1:4], dt)  
        quats[i,:] = QuaternionProduct(quat, quats[i-1,:])   # R_t = delta R_t * R_t-1
        quats[i,:] = quats[i,:] / LA.norm(quats[i,:]) 
    return quats 

def QuaternionReciprocal(q):
    quat = np.array([-q[0], -q[1], -q[2], q[3]])  
    return norm_quat(quat)

def torch_QuaternionReciprocal(q,  USE_CUDA = True):
    quat = torch.cat((-q[:,0:1], -q[:,1:2], -q[:,2:3], q[:,3:]), dim = 1) 
    batch_size = quat.size()[0]

    quat = torch_norm_quat(quat)
    return quat

def ProcessGyroData(gyro_data):
    quats = ProcessGyroRotation(gyro_data)
    size = np.shape(gyro_data)[0]
    axis_dif = np.zeros((size,3)) 
    for i in range(1, size):
        quat_dif = QuaternionProduct(quats[i,:], QuaternionReciprocal(quats[i-1,:]))  
        [axis_dif_cur, angles_cur] = ConvertQuaternionToAxisAngle(quat_dif)  
        axis_dif[i,:] = axis_dif_cur*angles_cur  
    return [axis_dif, quats]


def SlerpWithDefault(q1, q2, t, q_default):
    t = max(min(t, 1.0), 0.0) 
    kEpsilon = 1e-6 
    kSlerpLinearThresh = 0.9995 
    
    q1 = q1/LA.norm(q1) 
    q2 = q2/LA.norm(q2) 

    if t < kEpsilon:  # too close start(q1)
        q3 = q1 
        return q3
    elif t > 1-kEpsilon:  # too close end(q2)
        q3 = q2 
        return q3

    dot_prodcut = np.sum(q1*q2)  # nei ji   but no divide norm length

    if abs(dot_prodcut) >= 1:
        q3= q_default 
        return q3
    elif abs(dot_prodcut) > kSlerpLinearThresh:  #cos value is close 1, so angle is close 0du, so it is linear,  https://blog.csdn.net/u012947821/article/details/17136443
        q3 = q1*(1-t) + q2*t 
        q3 = q3/LA.norm(q3)     
        return q3

    sign = 1 
    if dot_prodcut < 0:
        sign = -1 
        dot_prodcut = -dot_prodcut 

    theta = np.arccos(dot_prodcut)
    # slert formula  https://blog.csdn.net/u012947821/article/details/17136443
    sin_theta = np.sin(theta) 
    inv_sin_theta = 1.0 / sin_theta 
    coeff1 = np.sin((1.0 - t) * theta) * inv_sin_theta 
    coeff2 = sign * np.sin(t * theta) * inv_sin_theta 
    q3 = q1 * coeff1 + q2 * coeff2 
    return q3


#Slerp
def GetGyroAtTimeStamp(gyro_data, timestamp):
    z = np.array([0,0,0,1])  
    if len(gyro_data) >= 2 and (not(timestamp < gyro_data[0,0] or timestamp > gyro_data[-1, 0])):
        ind = np.where(gyro_data[:,0] >= timestamp)
        ind = np.squeeze( ind, axis = 0)
        if gyro_data[ind[0], 0] == timestamp:
            z = gyro_data[ind[0],1:]
        else:
            start_index = ind[0] -1 
            end_index = ind[0] 
            ratio = (timestamp - gyro_data[start_index,0])/(gyro_data[end_index,0]-gyro_data[start_index,0])
            z = SlerpWithDefault(gyro_data[start_index,1:], gyro_data[end_index, 1:], ratio, gyro_data[start_index,1:]) 
    z = z / (LA.norm(z) + 1e-6)
    return z

def train_GetGyroAtTimeStamp(gyro_data, timestamp, check = False):
    if len(gyro_data) >= 2 and (not(timestamp < gyro_data[0,0] or timestamp > gyro_data[-1, 0])):
        ind = np.where(gyro_data[:,0] >= timestamp)
        ind = np.squeeze( ind, axis = 0)
        if gyro_data[ind[0], 0] == timestamp:
            z = gyro_data[ind[0],1:]
        else:
            start_index = ind[0] -1 
            end_index = ind[0] 
            ratio = (timestamp - gyro_data[start_index,0])/(gyro_data[end_index,0]-gyro_data[start_index,0])
            z = SlerpWithDefault(gyro_data[start_index,1:], gyro_data[end_index, 1:], ratio, gyro_data[start_index,1:]) 
        return z / (LA.norm(z) + 1e-6)
    if check:
        print("bad value")
    return None

def FindOISAtTimeStamp(ois_log, time):
    ois_time = ois_log[:,2] 
    if time <= ois_time[0]:
        ois_data = ois_log[0, 0:2] 
    elif time > ois_time[-1]:
        ois_data = ois_log[-1, 0:2]
    else:
        ind = np.where(ois_time >= time)
        ind = np.squeeze( ind, axis = 0)
        first_ind = ind[0]
        if ois_time[first_ind] == ind[0]:
            ois_data = ois_log[first_ind, 0:2]
        else:
            cur_time = ois_time[first_ind] 
            last_timestamp = ois_time[first_ind - 1]
            ratio = (time - last_timestamp) / (cur_time - last_timestamp) 
            ois_data = ois_log[first_ind - 1,0:2] * (1-ratio) + ois_log[first_ind,0:2]*ratio 

    return ois_data

def GetMetadata(frame_data, frame_index, result_poses = {} ):
    # global static_options
    # We can just use 1.27 as fov and virtual fov for videos in the data set.
    metadata = {}
    metadata["frame_id"] = frame_index
    metadata["timestamp_ns"]  = frame_data[frame_index, 0]
    metadata["timestamp_ois_ns"]  = frame_data[frame_index, 4]
    metadata["rs_time_ns"]  = frame_data[frame_index, 3]
    if "real fov" in result_poses:
        metadata["fov"] = result_poses['real fov'][frame_index,:] 
    else:
        metadata["fov"] = 1.27
    if "virtual fov" in result_poses:
        metadata["virtual_fov"] = result_poses['virtual fov'][frame_index,:] 
    else:
        metadata["virtual_fov"] = 1.27

    return metadata

def GetProjections(static_options, metadata, quats_data, ois_data,  no_shutter = False):
    num_rows = static_options["num_grid_rows"]
    real_projections = []
    for i in range(num_rows):
        if no_shutter:
            timestmap_ns = metadata["timestamp_ns"] + metadata["rs_time_ns"] * 0.5
            timestamp_ois_ns = metadata["timestamp_ois_ns"] + metadata["rs_time_ns"] * 0.5
        else:
            timestmap_ns = metadata["timestamp_ns"] + metadata["rs_time_ns"] * i / (num_rows-1)
            timestamp_ois_ns = metadata["timestamp_ois_ns"] + metadata["rs_time_ns"] * i / (num_rows-1)
        real_projections.append(GetRealProjection(
            static_options, quats_data, ois_data, metadata["fov"], timestmap_ns, timestamp_ois_ns))
    return real_projections

def GetRealProjection(static_options, quats_data, ois_data, fov, timestamp_ns, timestamp_ois_ns):
    quat = GetGyroAtTimeStamp(quats_data, timestamp_ns)
    ois_offset = FindOISAtTimeStamp(ois_data, timestamp_ois_ns) 
    # ois is w.r.t. active array size, thus we need to convert it to normalzied space.

    ois_offset = np.array(ois_offset) / np.array([static_options["crop_window_width"], static_options["crop_window_height"]])
    
    projection = GetProjectionHomography(quat, fov, ois_offset, static_options["width"], static_options["height"])
    return projection

def GetProjectionHomography(rot, fov, offset, width, height):
    # rot: rotation in quaternion
    # fov: sensor_width / focal_length.
    # offset: additional ois offset at normalized domain.
    # width/height: frame size.
    focal_length = width / fov
    rotation = ConvertQuaternionToRotationMatrix(rot)
    intrinsics = GetIntrinsics(focal_length, offset, width, height)
    projection_homography = np.matmul(intrinsics, rotation)
    return projection_homography

def torch_GetProjectionHomography(rot, fov, width, height, USE_CUDA = True):
    # rot: rotation in quaternion
    # fov: sensor_width / focal_length.
    # offset: additional ois offset at normalized domain.
    # width/height: frame size.
    focal_length = width / fov
    rotation = torch_ConvertQuaternionToRotationMatrix(rot)
    batch_size = rotation.size()[0]
    offset = np.array([0,0])
    intrinsics = GetIntrinsics(focal_length, offset, width, height)
    intrinsics = torch.Tensor(np.repeat(np.expand_dims(intrinsics, axis = 0), batch_size, axis = 0))
    if USE_CUDA == True:
        intrinsics = intrinsics.cuda()
    projection_homography = torch.matmul(intrinsics, rotation)
    return projection_homography

def ConvertQuaternionToRotationMatrix(quat):
    x = quat[0]
    y = quat[1]
    z = quat[2]
    w = quat[3]
    rotation = np.zeros(9)
    #https: // blog.csdn.net / loongkingwhat / article / details / 88427828            (formula 3)
    rotation[0] = 1 - 2 * y * y - 2 * z * z
    rotation[1] = 2 * x * y - 2 * z * w
    rotation[2] = 2 * x * z + 2 * y * w
    rotation[3] = 2 * x * y + 2 * z * w
    rotation[4] = 1 - 2 * x * x - 2 * z * z
    rotation[5] = 2 * y * z - 2 * x * w
    rotation[6] = 2 * x * z - 2 * y * w
    rotation[7] = 2 * y * z + 2 * x * w
    rotation[8] = 1 - 2 * x * x - 2 * y * y
    rotation = np.reshape(rotation, (3, 3)) # Note reshape is different with matlab
    return rotation

def torch_ConvertQuaternionToRotationMatrix(quat, USE_CUDA = True):
    x = quat[:,0]
    y = quat[:,1]
    z = quat[:,2]
    w = quat[:,3]

    batch_size = quat.size()[0]
    rotation = Variable(torch.zeros((batch_size, 9), requires_grad=True))
    if USE_CUDA == True:
        rotation = rotation.cuda()

    rotation[:,0] = 1 - 2 * y * y - 2 * z * z
    rotation[:,1] = 2 * x * y - 2 * z * w
    rotation[:,2] = 2 * x * z + 2 * y * w
    rotation[:,3] = 2 * x * y + 2 * z * w
    rotation[:,4] = 1 - 2 * x * x - 2 * z * z
    rotation[:,5] = 2 * y * z - 2 * x * w
    rotation[:,6] = 2 * x * z - 2 * y * w
    rotation[:,7] = 2 * y * z + 2 * x * w
    rotation[:,8] = 1 - 2 * x * x - 2 * y * y
    rotation = rotation.view(batch_size, 3, 3) # Note reshape is different with matlab
    return rotation

def ConvertRotationMatrixToQuaternion(m):
    tr = m[0,0] + m[1,1] + m[2,2]
    if tr > 0 :
        S = 2 * (tr+1.0)**0.5
        qw = 0.25 * S
        qx = (m[2,1] - m[1,2]) / S
        qy = (m[0,2] - m[2,0]) / S
        qz = (m[1,0] - m[0,1]) / S
    elif m[0,0] > m[1,1] and m[0,0] > m[2,2]:
        S = 2* (1.0 + m[0,0] - m[1,1] - m[2,2]) ** 0.5
        qw = (m[2,1] - m[1,2]) / S
        qx = 0.25 * S
        qy = (m[0,1] + m[1,0]) / S
        qz = (m[0,2] + m[2,0]) / S
    elif m[1,1] > m[2,2]: 
        S = 2* (1.0 - m[0,0] + m[1,1] - m[2,2]) ** 0.5
        qw = (m[0,2] - m[2,0]) / S
        qx = (m[0,1] + m[1,0]) / S
        qy = 0.25 * S
        qz = (m[1,2] + m[2,1]) / S
    else: 
        S = 2* (1.0 - m[0,0] - m[1,1] + m[2,2]) ** 0.5
        qw = (m[1,0] - m[0,1]) / S
        qx = (m[0,2] + m[2,0]) / S
        qy = (m[1,2] + m[2,1]) / S
        qz = 0.25 * S
    return np.array([qx,qy,qz,qw])

def GetIntrinsics(focal_length, offset, width, height):
    intrinsics = [
        [float(focal_length), 0.0, 0.5*(width-1)+offset[0]*width], 
        [0.0, float(focal_length), 0.5*(height-1)+offset[1]*height], 
        [0.0, 0.0, 1.0]
        ]
    return np.array(intrinsics)


def GetVirtualProjection(static_options, result_pose, metadata, frame_index):
    # debug only, for getting results and references for comparisons.
    quat = result_pose['virtual pose'][frame_index,:]
    if 'vitual lens offset' in result_pose:
        virutal_lens_offset = result_pose['vitual lens offset'][frame_index,:]
    else:
        virutal_lens_offset = np.array([0,0])
    virtual_projection = GetProjectionHomography(
        quat, metadata["virtual_fov"], virutal_lens_offset, static_options["width"], static_options["height"])
    return virtual_projection

def torch_GetVirtualProjection(static_options, quat, virtual_fov = 1.27):
    virtual_projection = torch_GetProjectionHomography(
        quat, virtual_fov, static_options["width"], static_options["height"])
    return virtual_projection


def GetForwardGrid(static_options, real_projections, virtual_projection):
    # real_projections: a set of 3x3 projections.
    # virtual_projection: a single 3x3 projection.

    grid = np.zeros((4, static_options["num_grid_cols"], static_options["num_grid_rows"]))
    width = static_options["width"]
    height = static_options["height"]

    row_step = 1/ (static_options["num_grid_rows"] - 1)
    col_step = 1/ (static_options["num_grid_cols"] - 1)

    for i in range(static_options["num_grid_rows"]):
        transform = GetHomographyTransformFromProjections(real_projections[i], virtual_projection)
        v = i * row_step
        for j in range(static_options["num_grid_cols"]):
            u = j * col_step
            point = np.array([u * width, v * height, 1]).T
            warped_point = ApplyTransform(transform, point)
            warped_point = warped_point / np.array([width, height, 1]) # normalize
            grid[:, j, i] = np.array([warped_point[0], warped_point[1], u, v])
    return grid

def torch_GetForwardGrid(static_options, real_projections, virtual_projection, USE_CUDA = True):
    # real_projections: a set of 3x3 projections.
    # virtual_projection: a single 3x3 projection.
    batch_size = real_projections.size()[0]

    grid = torch.zeros((batch_size, 4, static_options["num_grid_cols"], static_options["num_grid_rows"]))
    if USE_CUDA:
        grid = grid.cuda()
    width = static_options["width"]
    height = static_options["height"]

    row_step = 1/ (static_options["num_grid_rows"] - 1)
    col_step = 1/ (static_options["num_grid_cols"] - 1)

    for i in range(static_options["num_grid_rows"]):
        transform = torch_GetHomographyTransformFromProjections(real_projections[:, i], virtual_projection)
        v = i * row_step
        for j in range(static_options["num_grid_cols"]):
            u = j * col_step
            point = torch.Tensor([u * width, v * height, 1])
            norm = torch.Tensor([width, height, 1])
            if USE_CUDA == True:
                point = point.cuda()
                norm = norm.cuda()
            warped_point = torch_ApplyTransform(transform, point)
            warped_point = warped_point / norm # normalize
            grid[:, 0, j, i] = warped_point[:,0]
            grid[:, 1, j, i] = warped_point[:,1]
            grid[:, 2, j, i] = u
            grid[:, 3, j, i] = v
    return grid

def GetWarpingFlow(real_projections_src, real_projections_dst, num_rows, num_cols, frame_width, frame_height):
    # num_rows: rows of the flow.
    # num_cols: cols of the flow.
    grid = np.zeros((4, num_cols, num_rows))

    row_step = 1/ (num_rows - 1)
    col_step = 1/ (num_cols - 1)

    for i in range(num_rows):
        transform = GetHomographyTransformFromProjections(real_projections_src[i], real_projections_dst[i])
        v = i * row_step
        for j in range(num_cols):
            u = j * col_step
            point = np.array([u * frame_width, v * frame_height, 1]).T
            warped_point = ApplyTransform(transform, point)
            warped_point = warped_point / np.array([frame_width, frame_height, 1]) # normalize
            grid[:, j, i] = np.array([warped_point[0], warped_point[1], u, v])
    return grid

def torch_GetWarpingFlow(static_options, real_projections_src, real_projections_dst, USE_CUDA = True):
    # real_projections: a set of 3x3 projections.
    # virtual_projection: a single 3x3 projection.
    batch_size = real_projections_src.size()[0]

    grid = torch.zeros((batch_size, 4, static_options["num_grid_cols"], static_options["num_grid_rows"]))
    if USE_CUDA:
        grid = grid.cuda()
    width = static_options["width"]
    height = static_options["height"]

    row_step = 1/ (static_options["num_grid_rows"] - 1)
    col_step = 1/ (static_options["num_grid_cols"] - 1)

    for i in range(static_options["num_grid_rows"]):
        transform = torch_GetHomographyTransformFromProjections(real_projections_src[:, i], real_projections_dst[:, i])
        v = i * row_step
        for j in range(static_options["num_grid_cols"]):
            u = j * col_step
            point = torch.Tensor([u * width, v * height, 1])
            norm = torch.Tensor([width, height, 1])
            if USE_CUDA == True:
                point = point.cuda()
                norm = norm.cuda()
            warped_point = torch_ApplyTransform(transform, point)
            warped_point = warped_point / norm # normalize
            grid[:, 0, j, i] = warped_point[:,0]
            grid[:, 1, j, i] = warped_point[:,1]
            grid[:, 2, j, i] = u
            grid[:, 3, j, i] = v
    return grid

def GetHomographyTransformFromProjections(proj_src, proj_dst):
    return np.matmul(proj_dst, LA.inv(proj_src))

def torch_GetHomographyTransformFromProjections(proj_src, proj_dst):
    return torch.matmul(proj_dst, torch.inverse(proj_src))

def ApplyTransform(transform, point):
    # Warps a 2D point ([x y 1]) using a homography transform.
    # Returns the warped 2D point ([warped_x, warped_y, 1]).
    z = np.matmul(transform, point)
    z = z / z[2]
    return z

def torch_ApplyTransform(transform, point):
    # Warps a 2D point ([x y 1]) using a homography transform.
    # Returns the warped 2D point ([warped_x, warped_y, 1]).
    z = torch.matmul(transform, point)
    z = z / z[:,2:]
    return z

def CenterZoom(grid, ratio):
    grid[:, 0:2, :, :]  = (grid[:, 0:2, :, :] - 0.5) * ratio + 0.5
    return grid

