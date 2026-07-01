# KV Cache 正确性：数值演示

**设定：** $d_{head}=2$，1个头，$W_q = W_k = W_v = I$，即 $Q=K=V=x$，缩放因子 $\sqrt{d_{head}}=\sqrt{2}\approx1.414$

$$
x_0=[1,2],\quad x_1=[3,1],\quad x_2=[2,3],\quad x_3=[4,1]
$$

---

## Decode Step 0（Prefill）：序列 $[t_0]$，用 $t_0$ 的输出预测 $t_1$

此步为 **Prefill**，past\_kv 为空，朴素推理和 KV Cache 行为完全一致。

### 朴素推理 / KV Cache（传入 $[t_0]$，取 $\text{output}[-1]$）

$$
Q = K = V = \begin{bmatrix}1&2\end{bmatrix}
$$

$$
\text{scores} = \frac{QK^T}{\sqrt{2}}
= \frac{[1,2][1,2]^T}{\sqrt{2}}
= \frac{[5]}{\sqrt{2}} = [3.536]
$$

无需因果 mask（只有一个 token，无未来位置可屏蔽）：

$$
\text{weights} = \text{softmax}([3.536]) = [1.000]
$$

$$
\text{output}_0 = 1.000 \times [1,2] = \boxed{[1.000,\ 2.000]}
$$

> past\_kv 更新：$\{K_0=[1,2],\ V_0=[1,2]\}$，供后续 decode 步使用。

---

## Decode Step 1：序列 $[t_0, t_1]$，用 $t_1$ 的输出预测 $t_2$

### 朴素推理（传入 $[t_0, t_1]$，取 $\text{output}[-1]$）

$$
Q = K = V = \begin{bmatrix}1&2\\3&1\end{bmatrix}
$$

$$
\text{scores} = \frac{QK^T}{\sqrt{2}}
= \frac{1}{\sqrt{2}}\begin{bmatrix}1&2\\3&1\end{bmatrix}\begin{bmatrix}1&3\\2&1\end{bmatrix}
= \frac{1}{\sqrt{2}}\begin{bmatrix}5&5\\5&10\end{bmatrix}
= \begin{bmatrix}3.536&3.536\\3.536&7.071\end{bmatrix}
$$

加因果 mask（上三角置 $-\infty$）：

$$
\text{scores\_masked} = \begin{bmatrix}3.536 & -\infty \\ 3.536 & 7.071\end{bmatrix}
$$

$$
\text{weights} = \text{softmax}(\text{scores\_masked},\ \text{dim}=-1)
= \begin{bmatrix}1.000 & 0 \\ 0.030 & 0.970\end{bmatrix}
$$

$$
\text{output} = \text{weights} \cdot V
= \begin{bmatrix}1.000 & 0 \\ 0.030 & 0.970\end{bmatrix}\begin{bmatrix}1&2\\3&1\end{bmatrix}
= \begin{bmatrix}1.000 & 2.000 \\ 2.940 & 1.030\end{bmatrix}
$$

取最后一行：$\text{output}_1 = \boxed{[2.940,\ 1.030]}$

### KV Cache（只传 $t_1$，past\_kv $= \{K_0=[1,2],\ V_0=[1,2]\}$）

$$
Q_1=[3,1],\quad K_1=[3,1]
\qquad
K_{full}=\begin{bmatrix}1&2\\3&1\end{bmatrix},\quad V_{full}=\begin{bmatrix}1&2\\3&1\end{bmatrix}
$$

$$
\text{scores} = \frac{[3,1]\begin{bmatrix}1&3\\2&1\end{bmatrix}}{\sqrt{2}}
= \frac{[5,\ 10]}{\sqrt{2}} = [3.536,\ 7.071]
\xrightarrow{\text{softmax}} [0.030,\ 0.970]
$$

$$
\text{output}_1 = 0.030\times[1,2] + 0.970\times[3,1] = \boxed{[2.940,\ 1.030]} \checkmark
$$

---

## Decode Step 2：序列 $[t_0, t_1, t_2]$，用 $t_2$ 的输出预测 $t_3$

### 朴素推理（传入 $[t_0, t_1, t_2]$，取 $\text{output}[-1]$）

$$
Q = K = V = \begin{bmatrix}1&2\\3&1\\2&3\end{bmatrix}
$$

$$
\text{scores} = \frac{QK^T}{\sqrt{2}}
= \frac{1}{\sqrt{2}}\begin{bmatrix}1&2\\3&1\\2&3\end{bmatrix}\begin{bmatrix}1&3&2\\2&1&3\end{bmatrix}
= \frac{1}{\sqrt{2}}\begin{bmatrix}5&5&8\\5&10&9\\8&9&13\end{bmatrix}
= \begin{bmatrix}3.536&3.536&5.657\\3.536&7.071&6.364\\5.657&6.364&9.192\end{bmatrix}
$$

加因果 mask：

$$
\text{scores\_masked} = \begin{bmatrix}3.536&-\infty&-\infty\\3.536&7.071&-\infty\\5.657&6.364&9.192\end{bmatrix}
$$

$$
\text{weights} = \begin{bmatrix}1.000&0&0\\0.030&0.970&0\\0.007&0.028&0.965\end{bmatrix}
$$

$$
\text{output} = \text{weights} \cdot V
= \begin{bmatrix}1.000&0&0\\0.030&0.970&0\\0.007&0.028&0.965\end{bmatrix}
\begin{bmatrix}1&2\\3&1\\2&3\end{bmatrix}
= \begin{bmatrix}1.000&2.000\\2.940&1.030\\2.014&2.930\end{bmatrix}
$$

取最后一行：$\text{output}_2 = \boxed{[2.014,\ 2.930]}$

### KV Cache（只传 $t_2$，past\_kv $= \{K_0,K_1,V_0,V_1\}$）

$$
Q_2=[2,3],\quad K_2=[2,3]
\qquad
K_{full}=\begin{bmatrix}1&2\\3&1\\2&3\end{bmatrix},\quad V_{full}=\begin{bmatrix}1&2\\3&1\\2&3\end{bmatrix}
$$

$$
\text{scores} = \frac{[2,3]\begin{bmatrix}1&3&2\\2&1&3\end{bmatrix}}{\sqrt{2}}
= \frac{[8,\ 9,\ 13]}{\sqrt{2}} = [5.657,\ 6.364,\ 9.192]
\xrightarrow{\text{softmax}} [0.007,\ 0.028,\ 0.965]
$$

$$
\text{output}_2 = 0.007\times[1,2]+0.028\times[3,1]+0.965\times[2,3] = \boxed{[2.014,\ 2.930]} \checkmark
$$

---

## Decode Step 3：序列 $[t_0,t_1,t_2,t_3]$，用 $t_3$ 的输出预测 $t_4$

### 朴素推理（传入 $[t_0,t_1,t_2,t_3]$，取 $\text{output}[-1]$）

$$
Q = K = V = \begin{bmatrix}1&2\\3&1\\2&3\\4&1\end{bmatrix}
$$

$$
\text{scores} = \frac{QK^T}{\sqrt{2}}
= \frac{1}{\sqrt{2}}\begin{bmatrix}1&2\\3&1\\2&3\\4&1\end{bmatrix}
\begin{bmatrix}1&3&2&4\\2&1&3&1\end{bmatrix}
= \frac{1}{\sqrt{2}}\begin{bmatrix}5&5&8&6\\5&10&9&13\\8&9&13&11\\6&13&11&17\end{bmatrix}
= \begin{bmatrix}3.536&3.536&5.657&4.243\\3.536&7.071&6.364&9.192\\5.657&6.364&9.192&7.778\\4.243&9.192&7.778&12.021\end{bmatrix}
$$

加因果 mask：

$$
\text{scores\_masked} = \begin{bmatrix}
3.536 & -\infty & -\infty & -\infty \\
3.536 & 7.071   & -\infty & -\infty \\
5.657 & 6.364   & 9.192   & -\infty \\
4.243 & 9.192   & 7.778   & 12.021
\end{bmatrix}
$$

$$
\text{weights} = \begin{bmatrix}
1.000 & 0     & 0     & 0     \\
0.030 & 0.970 & 0     & 0     \\
0.007 & 0.028 & 0.965 & 0     \\
0.000 & 0.007 & 0.001 & 0.991
\end{bmatrix}
$$

$$
\text{output} = \text{weights} \cdot V
= \begin{bmatrix}
1.000 & 0     & 0     & 0     \\
0.030 & 0.970 & 0     & 0     \\
0.007 & 0.028 & 0.965 & 0     \\
0.000 & 0.007 & 0.001 & 0.991
\end{bmatrix}
\begin{bmatrix}1&2\\3&1\\2&3\\4&1\end{bmatrix}
= \begin{bmatrix}
1.000 & 2.000 \\
2.940 & 1.030 \\
2.014 & 2.930 \\
3.980 & 1.010
\end{bmatrix}
$$

取最后一行：$\text{output}_3 = \boxed{[3.980,\ 1.010]}$

### KV Cache（只传 $t_3$，past\_kv $= \{K_0,K_1,K_2,V_0,V_1,V_2\}$）

$$
Q_3=[4,1],\quad K_3=[4,1]
\qquad
K_{full}=\begin{bmatrix}1&2\\3&1\\2&3\\4&1\end{bmatrix},\quad V_{full}=\begin{bmatrix}1&2\\3&1\\2&3\\4&1\end{bmatrix}
$$

$$
\text{scores} = \frac{[4,1]\begin{bmatrix}1&3&2&4\\2&1&3&1\end{bmatrix}}{\sqrt{2}}
= \frac{[6,\ 13,\ 11,\ 17]}{\sqrt{2}} = [4.243,\ 9.192,\ 7.778,\ 12.021]
\xrightarrow{\text{softmax}} [0.000,\ 0.007,\ 0.001,\ 0.991]
$$

$$
\text{output}_3 = 0.000\times[1,2]+0.007\times[3,1]+0.001\times[2,3]+0.991\times[4,1] = \boxed{[3.980,\ 1.010]} \checkmark
$$

---

## 汇总对比

| 步骤 | 朴素：传入 | 朴素：Q/K/V 矩阵 | 取哪行 | KV Cache：传入 | output |
|------|----------|----------------|--------|---------------|--------|
| step 1 | $[t_0,t_1]$ | $2\times2$ | $[-1]$ | $t_1$ + past$\{K_0\}$ | $[2.940,\ 1.030]$ |
| step 2 | $[t_0,t_1,t_2]$ | $3\times2$ | $[-1]$ | $t_2$ + past$\{K_0,K_1\}$ | $[2.014,\ 2.930]$ |
| step 3 | $[t_0,t_1,t_2,t_3]$ | $4\times2$ | $[-1]$ | $t_3$ + past$\{K_0,K_1,K_2\}$ | $[3.980,\ 1.010]$ |

朴素每步都重算完整的 $Q,K,V$ 矩阵，然后用 `logits[-1]` 取最后一行；KV Cache 每步只算一行新 token 的 $Q,K,V$，从 past\_kv 拼出完整的 $K_{full},V_{full}$——两者对应行数值完全相同。
