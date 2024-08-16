# Latest with gram_schimit but too slow.

class SAXS_to_Eigenvalue(nn.Module):
    def __init__(self, input_shape, hidden_features, output_dim):
        super().__init__()
        self.channels=hidden_features
        self.q = nn.Linear(input_shape, hidden_features)
        self.k = nn.Linear(1, hidden_features)
        self.v = nn.Linear(1, hidden_features)
        self.out_layer = nn.Linear(hidden_features, output_dim)
    def forward(self, x):
        h_ = x[:, :, np.newaxis]
        q = self.q(h_.permute(0,2,1))
        print(q.device)
        k = self.k(h_)
        v = self.v(h_)
        w_ = torch.bmm(q,k.permute(0,2,1))
        w_ = w_ * (self.channels**(-0.5))
        w_ = torch.nn.functional.softmax(w_,dim=2)
        h_ = torch.bmm(w_,v)
        h_ = self.out_layer(h_)
        h_ = nn.ReLU()(h_)
        h_ = h_.squeeze(dim=1)
        return h_
    
class SAXS_to_Eigenvector_Cov(nn.Module):
    def __init__(self, input_shape, hidden_features, output_dim):
        super().__init__()
        self.channels=hidden_features
        self.q = nn.Conv1d(in_channels=1, out_channels=hidden_features, kernel_size=1,device='cuda')
        self.k = nn.Conv1d(in_channels=1, out_channels=hidden_features, kernel_size=1,device='cuda')
        self.v = nn.Conv1d(in_channels=1, out_channels=hidden_features, kernel_size=1,device='cuda')
        self.out_layer = nn.Linear(hidden_features, output_dim)
        self.out_layer_2 = nn.Linear(input_shape, output_dim)
        
    def gram_schmidt(self, vv):
        def projection(u, v):
            return (v * u).sum() / (u * u).sum() * u
        batch_size = vv.size(0)
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)
        for i in range(batch_size):
            ui = vv[i].clone()
            uu[i, :, 0] = ui[:, 0].clone()
            for k in range(1, nk):
                vk = vv[i, k].clone()
                uk = 0
                for j in range(0, k):
                    uj = uu[i, :, j].clone()
                    uk = uk + projection(uj, vk)
                uu[i, :, k] = vk - uk
            for k in range(nk):
                uk = uu[i, :, k].clone()
                uu[i, :, k] = uk / uk.norm()
        return uu

    def forward(self, x):
        h_ = x[:, np.newaxis, :]
        #print(x.shape, h_.shape)
        #h_ = self.upscale(h_)
        time_single=time.time()
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        time_single_end=time.time()
        print('time_single:', time_single_end-time_single)
        #print(k.shape)
        w_ = torch.bmm(q.permute(0,2,1),k)
        w_ = w_ * (self.channels**(-0.5))
        w_ = torch.nn.functional.softmax(w_,dim=2)
        #print(w_.shape)
        #print(v.shape)
        h_ = torch.bmm(w_,v.permute(0,2,1)) 
        print(h_.device) 
        #print(h_.shape)
        h_ = self.out_layer(h_)
        #print(h_.shape)
        h_ = self.out_layer_2(h_.permute(0,2,1))
        #print(h_.shape)
        time_grad = time.time()
        h_ = self.gram_schmidt(h_)
        time_grad_end = time.time()
        print('time_grad:', time_grad_end-time_grad)
        return h_
    
class HarmonicPrior(nn.Module):
    def __init__(self, input_shape, hidden_features, output_dim):
        super().__init__()
        self.a =3/(3.8**2)
        self.input_shape=input_shape
        self.channels=hidden_features
        self.output_dim=output_dim
        self.eigenvalue=SAXS_to_Eigenvalue(input_shape,hidden_features, output_dim)
        self.eigenvector=SAXS_to_Eigenvector_Cov(input_shape,hidden_features, output_dim)

    def forward(self, x):
        start_time=time.time()
        lambda_value=self.eigenvalue(x)
        self.lambda_value = torch.clamp(lambda_value, min=0.01)
        step1_time=time.time()
        nu_vector=self.eigenvector(x)
        self.nu_vector=nu_vector
        step2_time = time.time()
        batch_dims=x.size(0)
        lambda_value_inverse = torch.sqrt(1/self.lambda_value)
        step3_time = time.time()
        rand=torch.randn(batch_dims, self.output_dim, 3, device=x.device)
        step4_time = time.time()
        dot_product = torch.einsum('ij,ijk->ijk', lambda_value_inverse ,rand )
        return_value=torch.bmm(nu_vector, dot_product)
        step5_time = time.time()
        print('step 1:', step1_time-start_time)
        print('step 2:', step2_time-step1_time)
        print('step 3:', step3_time-step2_time)
        print('step 4:', step4_time-step3_time)
        print('step 5:', step5_time-step4_time)
        return return_value, torch.einsum('ij,ijk->ijk', self.lambda_value, self.nu_vector)
    
class PriorLoss(nn.Module):
    def __init__(self, N=256, a =3/(3.8**2)):
        super().__init__()
        self.a = a
        self.N = N
        self.background = self.fixed_background()
        self.mask_matrix = self.mask()
        self.loss_fn=nn.MSELoss(reduction='sum')

    def fixed_background(self):
        N = self.N
        J = torch.zeros(N, N)
        for i, j in zip(np.arange(N-1), np.arange(1, N)):
            J[i,i] += self.a
            J[j,j] += self.a
            J[i,j] = J[j,i] = - self.a
        return J
    
    def mask(self):
        diag_mask = torch.eye(self.N, dtype=torch.bool)
        superdiagonal_mask = torch.roll(diag_mask, shifts=1, dims=1)
        superdiagonal_mask[:, 0] = 0
        subdiagonal_mask = torch.roll(diag_mask, shifts=-1, dims=1)
        subdiagonal_mask[:, -1] = 0
        return diag_mask+superdiagonal_mask+subdiagonal_mask

    def forward(self, x):
        mask_matrix = self.mask_matrix.to(x.device)  # Ensure mask is on the same device as x
        background = self.background.unsqueeze(0).repeat(x.size(0), 1, 1).to(x.device)  # Ensure background is on the same device as x
        masked_x = x * mask_matrix.float()  # Apply mask
        return self.loss_fn(masked_x, background) # Compute and return the loss
    
#Mask version not complete: 

'''
class HarmonicPrior(nn.Module):
    def __init__(self, input_shape, hidden_features, output_dim):
        super().__init__()
        self.a =3/(3.8**2)
        self.channels=hidden_features
        self.output_dim=output_dim
        self.eigenvalue=SAXS_to_Eigenvalue(input_shape,hidden_features, output_dim)
        self.eigenvector=SAXS_to_Eigenvector_Cov(input_shape,hidden_features, output_dim)

    def diag_mask(self):
        diag_mask = torch.eye(self.output_dim, dtype=torch.bool)
        diag_mask[:, 0] = 0
        diag_mask[:,-1] = 0
        diag_mask_2 = torch.zeros((256, 256), dtype=torch.bool)
        diag_mask_2[0, 0] = 1
        diag_mask_2[255, 255] = 1
        return diag_mask, diag_mask_2
    
    def superdiagonal_mask(self,diag_mask):
        superdiagonal_mask = torch.roll(diag_mask, shifts=1, dims=1)
        superdiagonal_mask[:, 0] = 0
        return superdiagonal_mask
    
    def subdiagonal_mask(self,diag_mask):
        subdiagonal_mask = torch.roll(diag_mask, shifts=-1, dims=1)
        subdiagonal_mask[:, -1] = 0
        return subdiagonal_mask

    def forward(self, x):
        lambda_value=self.eigenvalue(x)
        nu_vector=self.eigenvector(x)
        result = torch.einsum('ij,ijk->ijk', lambda_value, nu_vector)
        
        batch_size = x.size(0)

        diag_mask, diag_mask_2 = self.diag_mask()
        superdiagonal_mask = self.superdiagonal_mask(diag_mask).unsqueeze(0).expand(batch_size, -1, -1)
        subdiagonal_mask = self.subdiagonal_mask(diag_mask).unsqueeze(0).expand(batch_size, -1, -1)
        diag_mask = diag_mask.unsqueeze(0).expand(batch_size, -1, -1)
        diag_mask_2 = diag_mask_2.unsqueeze(0).expand(batch_size, -1, -1)

        result[diag_mask] = 2 * self.a
        result[diag_mask_2] = self.a
        result[superdiagonal_mask] = -self.a
        result[subdiagonal_mask] = -self.a

        return result
'''

'''
class HarmonicPrior(nn.Module):
    def __init__(self, hidden_features, output_dim=256):
        super().__init__()
        self.channels=hidden_features
        self.q = nn.Linear(1, hidden_features)
        self.k = nn.Linear(1, hidden_features)
        self.v = nn.Linear(1, hidden_features)
        self.out_layer = nn.Linear(hidden_features, output_dim)
        self.orthognal_vector = nn.utils.parametrizations.orthogonal(nn.Linear(output_dim,output_dim))
        self.background = Fixed_Prior()
    def forward(self, x):
        h_ = x[:, :, np.newaxis]
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        w_ = torch.bmm(q,k.permute(0,2,1))
        w_ = w_ * (self.channels**(-0.5))
        w_ = torch.nn.functional.softmax(w_,dim=2)
        h_ = torch.bmm(w_,v)
        h_ = self.out_layer(h_)
        h_ = torch.matmul(x.unsqueeze(1),h_)
        h_ = h_.squeeze(1)
        h_ = nn.ReLU()(h_)
        h_inv = 1/h_
        h_inv[0] = 0 
        Q = self.orthognal_vector.weight
        return torch.matmul(Q,torch.sqrt(h_inv).T).T
        return torch.matmul(Q,torch.sqrt(h_inv).T).T + self.background.fixed_background()
'''

#Oldest, Bowen's version

'''
class HarmonicPrior:
    def __init__(self, N = 256, a =3/(3.8**2)):
        J = torch.zeros(N, N)
        for i, j in zip(np.arange(N-1), np.arange(1, N)):
            J[i,i] += a
            J[j,j] += a
            J[i,j] = J[j,i] = -a
        D, P = torch.linalg.eigh(J)
        D_inv = 1/D
        D_inv[0] = 0
        self.P, self.D_inv = P, D_inv
        self.N = N

    def to(self, device):
        self.P = self.P.to(device)
        self.D_inv = self.D_inv.to(device)
        
    def sample(self, batch_dims=()):
        return self.P @ (torch.sqrt(self.D_inv)[:,None] * torch.randn(*batch_dims, self.N, 3, device=self.P.device))
'''
    
'''
def transition_matrix(N_bins=1000, X_max=5):
    bins = torch.linspace(0, X_max, N_bins+1, dtype=torch.float64)
    cbins = (bins[1:] + bins[:-1]) / 2
    bw = cbins[1] - cbins[0]
    mu = 2 / cbins - cbins
    idx = torch.arange(N_bins)
    mat = torch.zeros((N_bins, N_bins), dtype=torch.float64)
    mat[idx, idx] = -2 / bw**2
    
    mat[idx[1:], idx[:-1]] = mu[idx[:-1]] / 2 / bw + 1 / bw**2    #   M_{i+1,i} = -mu[i]/2
    mat[idx[:-1], idx[1:]] = -mu[idx[1:]] / 2 / bw + 1 / bw**2    #   M_{i+1,i} = mu[i]/2
    mat[idx, idx] -= mat.sum(0) # fix edges

    return mat, bins

_mat, _bins = transition_matrix()
_D, _Q = torch.linalg.eig(_mat)
_Q_inv = torch.linalg.inv(_Q)
_sigmas = torch.from_numpy(np.load('chain_stats.npy'))


def add_noise(dists, residue_index, mask, t, device='cpu'):

    sigmas, Q, D, Q_inv, bins = _sigmas.to(device), _Q.to(device), _D.to(device), _Q_inv.to(device), _bins.to(device)
    
    mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
    # dists = torch.sum((pseudo_beta[..., None, :] - pseudo_beta[..., None, :, :]) ** 2, dim=-1)**0.5
    offsets = torch.abs(residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2))
    sigmas = sigmas[offsets]
    ndists = dists / sigmas * mask
    
    bindists = (ndists.unsqueeze(-1) > bins).sum(-1)
    bindists = torch.clamp(bindists, 0, 999)
    
    P = ((Q*torch.exp(D*t)) @ Q_inv).T # now we have a row stochatic matrix  P_ij = P(i -> j)
    probs = P.real[bindists]   # this is equivalent to left multiplication by basis e_i

    probs = torch.clamp(probs / probs.sum(-1, keepdims=True), 0, 1)
    newbindists = Categorical(probs, validate_args=False).sample()
    cbins = (bins[1:] + bins[:-1]) / 2
    newdists = cbins[newbindists] * mask * sigmas

    return newdists.float()

def sample_posterior(orig_dists, noisy_dists, residue_index, mask, s, t, device='cpu'):
    sigmas, Q, D, Q_inv, bins = _sigmas.to(device), _Q.to(device), _D.to(device), _Q_inv.to(device), _bins.to(device)
    mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)

    P_0s = ((Q*torch.exp(D*s)) @ Q_inv).T
    P_st = ((Q*torch.exp(D*(t-s))) @ Q_inv).T

    offsets = torch.abs(residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2))
    sigmas = sigmas[offsets]
    
    orig_ndists = orig_dists / sigmas * mask
    orig_bindists = (orig_ndists.unsqueeze(-1) > bins).sum(-1)
    orig_bindists = torch.clamp(orig_bindists, 0, 999)

    noisy_ndists = noisy_dists / sigmas * mask
    noisy_bindists = (noisy_ndists.unsqueeze(-1) > bins).sum(-1)
    noisy_bindists = torch.clamp(noisy_bindists, 0, 999)

    probs = P_0s.real[orig_bindists] * P_st.T.real[noisy_bindists]
    probs = torch.clamp(probs / probs.sum(-1, keepdims=True), 0, 1)
    newbindists = Categorical(probs, validate_args=False).sample()
    cbins = (bins[1:] + bins[:-1]) / 2
    newdists = cbins[newbindists] * mask * sigmas
    
    return newdists.float()

def sample_prior(residue_index, device='cpu'):

    sigmas, Q, D, Q_inv, bins = _sigmas.to(device), _Q.to(device), _D.to(device), _Q_inv.to(device), _bins.to(device)
    B, L = residue_index.shape
    probs = Q[:,D.real.argmax()].real
    probs = torch.clamp(probs / probs.sum(-1, keepdims=True), 0, 1).broadcast_to(B, L, L, 1000)
    
    offsets = torch.abs(residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2))
    sigmas = sigmas[offsets]
    
    newbindists = Categorical(probs, validate_args=False).sample()
       
    cbins = (bins[1:] + bins[:-1]) / 2
    newdists = cbins[newbindists] * sigmas
    return newdists.float()
'''

'''
    class PriorLoss(nn.Module):
    def __init__(self, N, a =3/(3.8**2)):
        super().__init__()
        self.a = a
        self.loss_fn=nn.MSELoss(reduction='sum')

    def fixed_background(self):
        N = self.N
        J = torch.zeros(256, 256)
        for i, j in zip(np.arange(N-1), np.arange(1, N)):
            #J[i,i] += self.a
            #J[j,j] += self.a
            J[i,j] = J[j,i] = - self.a
        return J
    # I should remove the diag_mask
    def mask(self):
        diag_mask = torch.eye(self.N, dtype=torch.float32)
        diag_mask = torch.nn.functional.pad(diag_mask, (0, 0, 0, diag_mask.size(0) - self.N))
        superdiagonal_mask = torch.roll(diag_mask, shifts=1, dims=1)
        superdiagonal_mask[:, 0] = 0
        subdiagonal_mask = torch.roll(diag_mask, shifts=-1, dims=1)
        subdiagonal_mask[:, -1] = 0
        return superdiagonal_mask+subdiagonal_mask
    
    def forward(self, x, N):
        self.N = N
        mask = self.mask()
        mask_matrix = mask.to(x.device)  # Ensure mask is on the same device as x
        background = self.fixed_background()
        background = background.unsqueeze(0).repeat(x.size(0), 1, 1).to(x.device)  # Ensure background is on the same device as x
        masked_x = x * mask_matrix.float()  # Apply mask
        return self.loss_fn(masked_x, background) # Compute and return the loss
'''
