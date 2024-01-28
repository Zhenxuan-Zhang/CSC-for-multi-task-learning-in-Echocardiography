from scipy.ndimage import morphology
def coefficients(gt, pred, smooth=1e-12):
    pred = torch.sigmoid(pred)
    pred = torch.gt(pred, 0.5)
    pred = pred.type(torch.float32)
    intersection = torch.sum(gt * pred)
    gt, pred = torch.sum(gt), torch.sum(pred)
    union = gt + pred - intersection

    precision = intersection / (pred + smooth)
    recall = intersection / (gt + smooth)

    beta_square = 0.3
    f_beta_coeff = (1 + beta_square) * precision * recall / (beta_square * precision + recall + smooth)
    dice_coeff = (2. * intersection) / (union + intersection + smooth)
    jaccard_coeff = intersection / (union + smooth)
    return dice_coeff, jaccard_coeff, f_beta_coeff

def acc(pred,gt):
    right,error = 0,0
    right_es,error_es = 0,0
    right_ed,error_ed = 0,0
    for i in range(30):
        if float(gt[i:i+1]) == 0:
            if float(gt[i:i+1]) == float(torch.max(pred[i:i+1], 1)[1]):
                    right_es = right_es+1
            else:
                    error_es = error_es+1
        elif float(gt[i:i+1]) == 1:
            if float(gt[i:i+1]) == float(torch.max(pred[i:i+1], 1)[1]):
                    right = right+1
            else:
                    error = error+1
        elif float(gt[i:i+1]) == 2:
            if float(gt[i:i+1]) == float(torch.max(pred[i:i+1], 1)[1]):
                    right_ed = right_ed+1
            else:
                    error_ed = error_ed+1
    return right/(right+error),right_es/(right_es+error_es+0.002),right_ed/(right_ed+error_ed+0.002)
    

def get_hausdorff(gt, pred, sampling=0.3, connectivity=1):
    pred = torch.sigmoid(pred)
    pred = torch.gt(pred, 0.5)
    input1 = gt
    input2 = pred
    input1 = np.array(input1.cpu().clone()) 
    input2 = np.array(input2.cpu().clone()) 
    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))

    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    S = input_1 ^ morphology.binary_erosion(input_1, conn)
    Sprime = input_2 ^ morphology.binary_erosion(input_2, conn)

    dta = morphology.distance_transform_edt(~S, sampling)
    dtb = morphology.distance_transform_edt(~Sprime, sampling)

    sds = np.concatenate([np.ravel(dta[Sprime != 0]), np.ravel(dtb[S != 0])])
    hausdorff_distance = sds.max()
    mean_abs_distance = np.abs(sds).mean()
    return hausdorff_distance, mean_abs_distance
    
def seg_loss(y_pred,y_true):
    y_pred = torch.sigmoid(y_pred)
    smooth       = 1e-12
    y_true_back  = 1 - y_true
    y_pred_back  = 1 - y_pred
    alpha        = 1 / (torch.pow(torch.sum(y_true), 2) + smooth)
    beta         = 1 / (torch.pow(torch.sum(y_true_back), 2) + smooth)
    numerater    = alpha * torch.sum(y_true * y_pred) + beta * torch.sum(y_true_back * y_pred_back)
    denominator  = alpha * torch.sum(y_true + y_pred) + beta * torch.sum(y_true_back + y_pred_back)
    dice_loss    = 1 - (2. * numerater) / (denominator + smooth)
    mae_loss     = torch.mean(torch.log(1 + torch.exp(torch.abs(y_pred - y_true))))
    w            = (img_size * img_size - torch.sum(y_pred)) / (torch.sum(y_pred) + smooth)
    key_w        = 0.003
    crossentropy = - torch.mean(key_w * w * y_true * torch.log(y_pred + smooth) + y_true_back * torch.log(y_pred_back + smooth))
    #print(crossentropy)
    return crossentropy + dice_loss + mae_loss

def one_hot(label, n_classes, requires_grad=True):
    """Return One Hot Label"""
    divce = label.device
    one_hot_label = torch.eye(n_classes, device=device, requires_grad=requires_grad)[label]
    # one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)
    return one_hot_label

def boundary_cos_loss(gt , pred):
    pred = torch.sigmoid(pred)
    #gt dimension (B,T,C,H,W)
    b,t,c,h,w = gt.shape
    for i in range(t):
        gt_frame = gt[0,i:i+1,:,:,:]
        pred_frame = pred[0,i:i+1,:,:,:]
        theta0 = 3
        gt_cont = F.max_pool2d(1 - gt_frame, kernel_size=theta0, stride=1, padding=(theta0 - 1) // 2)
        gt_cont -= 1 - gt_frame
        pred_cont = F.max_pool2d(1 - pred_frame, kernel_size=theta0, stride=1, padding=(theta0 - 1) // 2)
        pred_cont -= 1 - pred_frame
        sim = torch.cosine_similarity(gt_cont.squeeze(0).squeeze(0),pred_cont.squeeze(0).squeeze(0))
        sim_norm = torch.sum(sim)/(h)
        sim_loss = 1 - sim_norm*2
        if i == 0:
            loss = sim_loss
        else:
            loss = loss + sim_loss
    return loss / t

def cos_sim_loss(gt , pred):
    pred = torch.sigmoid(pred)
    #gt dimension (B,T,C,H,W)
    b,t,c,h,w = gt.shape
    for i in range(t):
        gt_frame = gt[0,i:i+1,:,:,:]
        pred_frame = pred[0,i:i+1,:,:,:]
        theta0 = 3
        sim = torch.cosine_similarity(gt_frame.squeeze(0).squeeze(0),pred_frame.squeeze(0).squeeze(0))
        sim_norm = torch.sum(sim)/(h)
        sim_loss = 1 - sim_norm*2
        if i == 0:
            loss = sim_loss
        else:
            loss = loss + sim_loss
    return loss / t

def corr_loss(pred,gt):
    pred_lva,pred_mvd,pred_lvd = pred[0,:],pred[1,:],pred[2,:]
    gt_lva,  gt_mvd,  gt_lvd   = gt[0,:],  gt[1,:],  gt[2,:]
    pred,gt = pred_lva, gt_lva
    pred_mean, gt_mean = torch.mean(pred), torch.mean(gt)
    corr_lva = (torch.sum((pred - pred_mean) * (gt - gt_mean))) / ((
                torch.sqrt(torch.sum((pred - pred_mean) ** 2)) * torch.sqrt(torch.sum((gt - gt_mean) ** 2)))+1e-12)
    pred,gt = pred_mvd, gt_mvd
    pred_mean, gt_mean = torch.mean(pred), torch.mean(gt)
    corr_mvd = (torch.sum((pred - pred_mean) * (gt - gt_mean))) / ((
                torch.sqrt(torch.sum((pred - pred_mean) ** 2)) * torch.sqrt(torch.sum((gt - gt_mean) ** 2)))+1e-12)
    pred,gt = pred_lvd, gt_lvd
    pred_mean, gt_mean = torch.mean(pred), torch.mean(gt)
    corr_lvd = (torch.sum((pred - pred_mean) * (gt - gt_mean))) / ((
                torch.sqrt(torch.sum((pred - pred_mean) ** 2)) * torch.sqrt(torch.sum((gt - gt_mean) ** 2)))+1e-12)
    #print('corr1:',corr_lva,'corr2:',corr_mvd,'corr3:',corr_lvd)
    corr = corr_lva+2*corr_mvd+2*corr_lvd+1e-12
    return 5-corr

def mae_loss(pred,gt):
    mae1 = torch.mean(torch.abs(pred[0,:]-gt[0,:]))
    mae2 = torch.mean(torch.abs(pred[1,:]-gt[1,:]))
    mae3 = torch.mean(torch.abs(pred[2,:]-gt[2,:]))
    mae = mae2+mae3*2
    #mae = torch.mean((pred-gt)* torch.tanh(pred-gt))
    #logcosh = torch.mean(torch.log(torch.cosh((pred-gt) + 1e-12)))
    mae_mean = torch.mean(torch.abs(torch.mean(pred)-torch.mean(gt)))
    return mae

def mae_point(pred,gt):
    mae = torch.mean(torch.abs(pred-gt))
    #mae = torch.mean((pred-gt)* torch.tanh(pred-gt))
    logcosh = torch.mean(torch.log(torch.cosh((pred-gt) + 1e-12)))
    return mae

def mae_cal(pred,gt):
    mae1 = torch.mean(torch.abs(pred[0,:]-gt[0,:]))
    mae2 = torch.mean(torch.abs(pred[1,:]-gt[1,:]))
    mae3 = torch.mean(torch.abs(pred[2,:]-gt[2,:]))
    mae = mae2+mae3*2
    #mae = torch.mean((pred-gt)* torch.tanh(pred-gt))
    logcosh = torch.mean(torch.log(torch.cosh((pred-gt) + 1e-12)))
    return mae


def person_corr(pred, gt):#皮尔森相关系数
    pred_lva,pred_mvd,pred_lvd = pred[0,:],pred[1,:],pred[2,:]
    gt_lva,  gt_mvd,  gt_lvd   = gt[0,:],  gt[1,:],  gt[2,:]
    pred,gt = pred_lva, gt_lva
    pred_mean, gt_mean = torch.mean(pred), torch.mean(gt)
    corr_lva = (torch.sum((pred - pred_mean) * (gt - gt_mean))) / (
                torch.sqrt(torch.sum((pred - pred_mean) ** 2)) * torch.sqrt(torch.sum((gt - gt_mean) ** 2))+1e-12)
    pred,gt = pred_mvd, gt_mvd
    pred_mean, gt_mean = torch.mean(pred), torch.mean(gt)
    corr_mvd = (torch.sum((pred - pred_mean) * (gt - gt_mean))) / (
                torch.sqrt(torch.sum((pred - pred_mean) ** 2)) * torch.sqrt(torch.sum((gt - gt_mean) ** 2))+1e-12)
    pred,gt = pred_lvd, gt_lvd
    pred_mean, gt_mean = torch.mean(pred), torch.mean(gt)
    corr_lvd = (torch.sum((pred - pred_mean) * (gt - gt_mean))) / (
                torch.sqrt(torch.sum((pred - pred_mean) ** 2)) * torch.sqrt(torch.sum((gt - gt_mean) ** 2))+1e-12)
    return corr_lva, corr_mvd,corr_lvd

def pot2index(srcnpy):
    mvd_gt,lvd_gt  = [],[]
    # print(srcnpy.shape)#1,30,4,2
    # for i in range(srcnpy.shape[0]):
    mvd1,lvd1 = point2linear(srcnpy[0])
    mvd_gt.append(mvd1)
    lvd_gt.append(lvd1)
    #print(lvd,lvd1)
    lvd_gt = np.array(lvd_gt)[:,:,np.newaxis]
    mvd_gt = np.array(mvd_gt)[:,:,np.newaxis]
    index  = np.concatenate((lvd_gt, mvd_gt), axis=2)
    return index

def mae_div(pred,gt):
    pred = pred.cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()
    pred_ind = pot2index(pred)
    gt_ind   = pot2index(gt)
    # print(gt_ind.shape)
    mae1 = np.mean(np.abs(pred_ind[:,:,0]-gt_ind[:,:,0]))
    mae2 = np.mean(np.abs(pred_ind[:,:,1]-gt_ind[:,:,1]))
    mae = mae1+mae2
    return mae1,mae2