import torch
import torch.nn
import triton
from einops import einsum, rearrange
import math
import triton.language as tl

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False, B_q=32, B_k=32):
        # Initialize parameters
        *batch_dims, seq_len, head_dim = Q.size()
        scale = head_dim ** (-0.5)
        T_q = math.ceil(seq_len / B_q)
        T_k = math.ceil(seq_len / B_k)
        O = torch.zeros_like(Q)
        # Compute log-sum-exp: Σ exp(S) = exp(m) * Σ exp(S - m) = exp(m) * l
        # log-sum-exp(x) = m + log( sum_j exp(x_j - m) ) = m + log(l)
        # This is the logarithm of the softmax denominator
        L_list = []
        
        # Traverse the query tile and initialize the state
        for i in range(T_q):
            Q_i = Q[..., i*B_q:(i+1)*B_q, :]
            # It's possible that the last chunk isn't full, 
            # Python will automatically handle the overflow, but we need to get the size of this chunk.
            B_q_actual = Q_i.size(-2)
            # initialize the state for online softmax
            # The current global maximum score at each query position.
            m_i = torch.full((*batch_dims, B_q_actual), float("-inf"), dtype=Q.dtype, device=Q.device)
            # The softmax denominator at each query position 
            # (sum of exponents, based on the latest max value)
            l_i = torch.zeros(*batch_dims, B_q_actual, dtype=Q.dtype, device=Q.device)
            # The sum of exp(attn_scores) * V
            O_i = torch.zeros_like(Q_i)
            
            for j in range(T_k):
                K_j = K[..., j*B_k:(j+1)*B_k, :]
                V_j = V[..., j*B_k:(j+1)*B_k, :]
                B_k_actual = K_j.size(-2)
                # S_ij = torch.matmul(Q_i, K_j.transpose(-1, -2)) * scale
                # S_ij = Q_i @ K_j.transpose(-1, -2) * scale
                S_ij = einsum(Q_i, K_j, "... B_q d_k, ... B_k d_k-> ... B_q B_k") * scale
                # causal mask
                if is_causal:
                    q_start = i * B_q
                    k_start = j * B_k
                    q_pos = torch.arange(q_start, q_start + B_q_actual, device=Q.device)
                    k_pos = torch.arange(k_start, k_start + B_k_actual, device=Q.device)
                    mask = q_pos.unsqueeze(-1) >= k_pos.unsqueeze(-2)
                    S_ij = S_ij.masked_fill(~mask, float("-inf"))
                    
                # online softmax
                m_ij = S_ij.max(dim=-1).values # find max in S_ij,shape = [B_q_actual]
                m_new = torch.max(m_i, m_ij)
                alpha = torch.exp(m_i - m_new)
                
                O_i = O_i * alpha.unsqueeze(-1) # [B_q_actual] -> [B_q_actual, B_head] * [B_q_actual, 1] -> [B_q_actual, B_head]
                l_i = l_i * alpha
                
                P_ij = torch.exp(S_ij - m_new.unsqueeze(-1))
                l_i += P_ij.sum(dim=-1)
                O_i += einsum(P_ij, V_j, "... B_q B_k, ... B_k d_k -> ... B_q d_k")
                m_i = m_new
            O_i = O_i / l_i.unsqueeze(-1)
            O[..., i*B_q:i*B_q + B_q_actual, :] = O_i

            L_i = m_i + torch.log(l_i)
            L_list.append(L_i)    
        L = torch.cat(L_list, dim=-1)
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.B_q = B_q
        ctx.B_k = B_k
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_output):
        L, Q, K, V, O = ctx.saved_tensors
        B_q = ctx.B_q
        B_k = ctx.B_k
        is_causal = ctx.is_causal
        *batch_dims, seq_len, d_head = Q.shape
        scale = d_head ** -0.5

        T_q = math.ceil(seq_len / B_q)
        T_k = math.ceil(seq_len / B_k)

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        
        for i in range(T_q):
            Q_i = Q[..., i*B_q : (i+1)*B_q, :]
            B_q_actual = Q_i.size(-2)
            
            dO_i = grad_output[..., i*B_q : (i+1)*B_q, :]
            O_i  = O[..., i*B_q : (i+1)*B_q, :]
            L_i  = L[..., i*B_q : (i+1)*B_q]           
            # log-sum-exp
            D = (dO_i * O_i).sum(dim=-1)
            # dS = P ⊙ (dP - D.unsqueeze(-1))
            # dQ_i = Σ (dS_ij @ K_j * scale)
            dQ_i = torch.zeros_like(Q_i)
            for j in range(T_k):
                K_j = K[..., j*B_k:(j+1)*B_k, :]
                V_j = V[..., j*B_k:(j+1)*B_k, :]
                B_k_actual = K_j.size(-2)

                S_ij = einsum(Q_i, K_j,
                        "... B_q d_k, ... B_k d_k -> ... B_q B_k") * scale
                
                # causal mask
                if is_causal:
                    q_start = i * B_q
                    k_start = j * B_k
                    q_pos = torch.arange(q_start, q_start + B_q_actual, device=Q.device)
                    k_pos = torch.arange(k_start, k_start + B_k_actual, device=Q.device)
                    mask = q_pos.unsqueeze(-1) >= k_pos.unsqueeze(-2)
                    S_ij = S_ij.masked_fill(~mask, float('-inf'))

                P_ij = torch.exp(S_ij - L_i.unsqueeze(-1))

                dV_j = einsum(P_ij, dO_i, "... B_q B_k, ... B_q d_k -> ... B_k d_k")
                dV[..., j*B_k : j*B_k + B_k_actual, :] += dV_j
                
                dP_ij = einsum(dO_i, V_j, "... B_q d_k, ... B_k d_k -> ... B_q B_k")
                dS_ij = P_ij * (dP_ij - D.unsqueeze(-1))
                
                dQ_i += einsum(dS_ij, K_j, "... B_q B_k, ... B_k d_k -> ... B_q d_k") * scale
                dK_j = einsum(dS_ij, Q_i, "... B_q B_k, ... B_q d_k -> ... B_k d_k") * scale
                dK[..., j*B_k : j*B_k + B_k_actual, :] += dK_j

            dQ[..., i*B_q : i*B_q + B_q_actual, :] = dQ_i
        return dQ, dK, dV, None, None, None

@triton.jit
def flash_attn_forward_kernel(
    # pointer
    q_ptr, k_ptr, v_ptr, o_ptr, l_ptr,
    # stride
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_h, stride_o_s, stride_o_d,
    stride_l_b, stride_l_h, stride_l_s, 
    # parameters
    seq_len, d_head,
    B_q, B_k, T_k,
    is_causal: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    # PyTorch: for i in range(T_q):
    pid_q = tl.program_id(2)

    # PyTorch: Q_i = Q[..., i*B_q:(i+1)*B_q, :]
    q_start = pid_q * B_q
    q_offsets = q_start + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < seq_len
    # 第几层（batch）第几区（head）
    q_base = q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    k_base = k_ptr + pid_b * stride_k_b + pid_h * stride_k_h
    v_base = v_ptr + pid_b * stride_v_b + pid_h * stride_v_h
    o_base = o_ptr + pid_b * stride_o_b + pid_h * stride_o_h
    l_base = l_ptr + pid_b * stride_l_b + pid_h * stride_l_h

    # m_i = torch.full((*batch_dims, B_q_actual), float("-inf"))
    m_i = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
    # l_i = torch.zeros(*batch_dims, B_q_actual, dtype=Q.dtype, device=Q.device)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    # O_i = torch.zeros_like(Q_i)
    acc = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)
    
    # q_base + 第几排（seq）第几个位（dim）
    # load Q
    Q = tl.load(
        # 标量 + 列向量 + 行向量 = 标量广播 + 2D矩阵
        q_base + q_offsets[:, None] * stride_q_s + tl.arange(0, BLOCK_D)[None, :] * stride_q_d,
        mask=q_mask[:, None] & (tl.arange(0, BLOCK_D)[None, :] < d_head)
    )

    for pid_k in range(T_k):
        k_start = pid_k * B_k
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < seq_len

        # 加载 K
        K = tl.load(
            k_base + k_offsets[:, None] * stride_k_s + tl.arange(0, BLOCK_D)[None, :] * stride_k_d,
            mask=k_mask[:, None] & (tl.arange(0, BLOCK_D)[None, :] < d_head)
        )
        
        # 加载 V
        V = tl.load(
            v_base + k_offsets[:, None] * stride_v_s + tl.arange(0, BLOCK_D)[None, :] * stride_v_d,
            mask=k_mask[:, None] & (tl.arange(0, BLOCK_D)[None, :] < d_head)
        )

        # calculate S = Q @ K^T
        S = tl.zeros([BLOCK_Q, BLOCK_K], dtype=tl.float32)

        for d in range(0, d_head, BLOCK_D):
            # 用 tl.arange 创建索引
            d_offsets = d + tl.arange(0, BLOCK_D)
            d_mask = d_offsets < d_head
            
            Q_chunk = tl.load(
                q_base + q_offsets[:, None] * stride_q_s + d_offsets[None, :] * stride_q_d,
                mask=q_mask[:, None] & d_mask[None, :]
            )
            K_chunk = tl.load(
                k_base + k_offsets[:, None] * stride_k_s + d_offsets[None, :] * stride_k_d,
                mask=k_mask[:, None] & d_mask[None, :]
            )
            S += tl.dot(Q_chunk, tl.trans(K_chunk))
        S *= 1.0 / tl.sqrt(d_head.to(tl.float32))
        if is_causal:
            q_pos = q_offsets[:, None]
            k_pos = k_offsets[None, :]
            S = tl.where(q_pos >= k_pos, S, float("-inf"))
        # online softmax
        m_ij = tl.max(S, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        m_i = m_new
        
        # acc  Σ exp(S-max) * V	 (BLOCK_Q, BLOCK_D)
        # l_i  Σ exp(S-max)	     (BLOCK_Q,)
        # acc 是加权和，l_i 是权重和，相除就是加权平均
        acc = acc * alpha[:, None]
        l_i = l_i * alpha

        P = tl.exp(S - m_new[:, None])
        l_i += tl.sum(P, axis=1)
        acc += tl.dot(P.to(V.dtype), V)


    O = acc / l_i[:, None]

    # 计算并存储 L
    L = m_i + tl.log(l_i)
    tl.store(
        l_base + q_offsets * stride_l_s,
        L,
        mask=q_mask
    )
    # 存储结果
    tl.store(
        o_base + q_offsets[:, None] * stride_o_s + tl.arange(0, BLOCK_D)[None, :] * stride_o_d,
        O,
        mask=q_mask[:, None] & (tl.arange(0, BLOCK_D)[None, :] < d_head)
    )

@triton.jit
def flash_attn_backward_kernel(
    # 指针
    q_ptr, k_ptr, v_ptr, o_ptr, do_ptr,
    dq_ptr, dk_ptr, dv_ptr,
    # stride
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_h, stride_o_s, stride_o_d,
    stride_do_b, stride_do_h, stride_do_s, stride_do_d,
    stride_dq_b, stride_dq_h, stride_dq_s, stride_dq_d,
    stride_dk_b, stride_dk_h, stride_dk_s, stride_dk_d,
    stride_dv_b, stride_dv_h, stride_dv_s, stride_dv_d,
    # 参数
    seq_len, d_head,
    B_q, B_k,
    T_k,
    is_causal: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)

    q_start = pid_q * B_q
    q_offsets = q_start + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < seq_len

    q_base = q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    k_base = k_ptr + pid_b * stride_k_b + pid_h * stride_k_h
    v_base = v_ptr + pid_b * stride_v_b + pid_h * stride_v_h
    o_base = o_ptr + pid_b * stride_o_b + pid_h * stride_o_h
    do_base = do_ptr + pid_b * stride_do_b + pid_h * stride_do_h
    dq_base = dq_ptr + pid_b * stride_dq_b + pid_h * stride_dq_h
    dk_base = dk_ptr + pid_b * stride_dk_b + pid_h * stride_dk_h
    dv_base = dv_ptr + pid_b * stride_dv_b + pid_h * stride_dv_h
    
    # 加载 Q
    Q = tl.load(
        q_base + q_offsets[:, None] * stride_q_s + tl.arange(0, BLOCK_D)[None, :] * stride_q_d,
        mask=q_mask[:, None] & (tl.arange(0, BLOCK_D)[None, :] < d_head)
    )
    
    # 加载 O
    O = tl.load(
        o_base + q_offsets[:, None] * stride_o_s + tl.arange(0, BLOCK_D)[None, :] * stride_o_d,
        mask=q_mask[:, None] & (tl.arange(0, BLOCK_D)[None, :] < d_head)
    )
    
    # 加载 dO（上游梯度）
    dO = tl.load(
        do_base + q_offsets[:, None] * stride_do_s + tl.arange(0, BLOCK_D)[None, :] * stride_do_d,
        mask=q_mask[:, None] & (tl.arange(0, BLOCK_D)[None, :] < d_head)
    )

    D = tl.sum(dO * O, axis=1)
    # 初始化 dQ_i
    dQ_i = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)
    
    # 遍历 key 块
    for pid_k in range(T_k):
        k_start = pid_k * B_k
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < seq_len
        
        # 加载 K
        K = tl.load(
            k_base + k_offsets[:, None] * stride_k_s + tl.arange(0, BLOCK_D)[None, :] * stride_k_d,
            mask=k_mask[:, None] & (tl.arange(0, BLOCK_D)[None, :] < d_head)
        )
        
        # 加载 V
        V = tl.load(
            v_base + k_offsets[:, None] * stride_v_s + tl.arange(0, BLOCK_D)[None, :] * stride_v_d,
            mask=k_mask[:, None] & (tl.arange(0, BLOCK_D)[None, :] < d_head)
        )
        
        # 计算 S = Q @ K^T
        S = tl.zeros([BLOCK_Q, BLOCK_K], dtype=tl.float32)
        for d in range(0, d_head, BLOCK_D):
            d_offsets = d + tl.arange(0, BLOCK_D)
            d_mask = d_offsets < d_head
            
            Q_chunk = tl.load(
                q_base + q_offsets[:, None] * stride_q_s + d_offsets[None, :] * stride_q_d,
                mask=q_mask[:, None] & d_mask[None, :]
            )
            K_chunk = tl.load(
                k_base + k_offsets[:, None] * stride_k_s + d_offsets[None, :] * stride_k_d,
                mask=k_mask[:, None] & d_mask[None, :]
            )
            S += tl.dot(Q_chunk, tl.trans(K_chunk))
        S *= 1.0 / tl.sqrt(d_head.to(tl.float32))
        
        # 因果掩码
        if is_causal:
            q_pos = q_offsets[:, None]
            k_pos = k_offsets[None, :]
            S = tl.where(q_pos >= k_pos, S, float("-inf"))
        
        # P = softmax(S)
        L_i = tl.logsumexp(S, axis=1)
        P = tl.exp(S - L_i[:, None])
        
        # dV_j = P^T @ dO
        dV_j = tl.dot(tl.trans(P).to(dO.dtype), dO)
        tl.atomic_add(
            dv_base + k_offsets[:, None] * stride_dv_s + tl.arange(0, BLOCK_D)[None, :] * stride_dv_d,
            dV_j,
            mask=k_mask[:, None] & (tl.arange(0, BLOCK_D)[None, :] < d_head)
        )
        
        # dP = dO @ V^T
        dP = tl.dot(dO, tl.trans(V))
        
        # dS = P * (dP - D[:, None])
        dS = P * (dP - D[:, None])
        
        if is_causal:
            q_pos = q_offsets[:, None]
            k_pos = k_offsets[None, :]
            dS = tl.where(q_pos >= k_pos, dS, 0.0)
        
        # dQ_i += dS @ K * scale
        dQ_i += tl.dot(dS, K) * (1.0 / tl.sqrt(d_head))

        # dK_j = dS^T @ Q * scale
        dK_j = tl.dot(tl.trans(dS), Q) * (1.0 / tl.sqrt(d_head))
        tl.atomic_add(
            dk_base + k_offsets[:, None] * stride_dk_s + tl.arange(0, BLOCK_D)[None, :] * stride_dk_d,
            dK_j,
            mask=k_mask[:, None] & (tl.arange(0, BLOCK_D)[None, :] < d_head)
        )
    
    # 存储 dQ
    tl.store(
        dq_base + q_offsets[:, None] * stride_dq_s + tl.arange(0, BLOCK_D)[None, :] * stride_dq_d,
        dQ_i,
        mask=q_mask[:, None] & (tl.arange(0, BLOCK_D)[None, :] < d_head)
    )


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False, B_q=32, B_k=32):
        if Q.dim() == 3:
            batch_size, seq_len, d_head = Q.shape
            num_heads = 1
            Q = Q.unsqueeze(1)
            K = K.unsqueeze(1)
            V = V.unsqueeze(1)
            squeeze_output = True
        elif Q.dim() == 4:
            batch_size, num_heads, seq_len, d_head = Q.shape
            squeeze_output = False
        else:
            raise ValueError(f"Q must be 3D or 4D, got {Q.dim()}D")
            
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
        
        T_q = (seq_len + B_q - 1) // B_q
        T_k = (seq_len + B_k - 1) // B_k
        
        O = torch.empty_like(Q)
        L = torch.empty(batch_size, num_heads, seq_len, dtype=Q.dtype, device=Q.device)

        BLOCK_Q = triton.next_power_of_2(B_q)
        BLOCK_K = triton.next_power_of_2(B_k)
        BLOCK_D = triton.next_power_of_2(d_head)

        grid = (batch_size, num_heads, T_q) # 3D grid

        flash_attn_forward_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            seq_len, d_head,
            B_q, B_k,
            T_k,
            is_causal,
            BLOCK_Q=BLOCK_Q,
            BLOCK_K=BLOCK_K,
            BLOCK_D=BLOCK_D,
        )
        
        ctx.save_for_backward(Q, K, V, O, L.squeeze(1) if squeeze_output else L)
        ctx.B_q = B_q
        ctx.B_k = B_k
        ctx.is_causal = is_causal
        ctx.squeeze_output = squeeze_output
        if squeeze_output:
            O = O.squeeze(1)
        return O
    
    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, O, L = ctx.saved_tensors
        B_q = ctx.B_q
        B_k = ctx.B_k
        is_causal = ctx.is_causal
        
        if ctx.squeeze_output:
            grad_output = grad_output.unsqueeze(1)
        
        batch_size, num_heads, seq_len, d_head = Q.shape
        
        T_q = (seq_len + B_q - 1) // B_q
        T_k = (seq_len + B_k - 1) // B_k
        
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        
        BLOCK_Q = triton.next_power_of_2(B_q)
        BLOCK_K = triton.next_power_of_2(B_k)
        BLOCK_D = triton.next_power_of_2(d_head)
        
        grid = (batch_size, num_heads, T_q)
        
        flash_attn_backward_kernel[grid](
            Q, K, V, O, grad_output,
            dQ, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            seq_len, d_head,
            B_q, B_k,
            T_k,
            is_causal,
            BLOCK_Q=BLOCK_Q,
            BLOCK_K=BLOCK_K,
            BLOCK_D=BLOCK_D,
        )
        
        if ctx.squeeze_output:
            dQ = dQ.squeeze(1)
            dK = dK.squeeze(1)
            dV = dV.squeeze(1)
        
        return dQ, dK, dV, None, None, None