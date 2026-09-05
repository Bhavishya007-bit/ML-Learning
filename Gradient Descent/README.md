# Gradient Descent for Linear Regression — Mathematical Notes

## Purpose

This README is a reference for understanding **Linear Regression with Gradient Descent from scratch**, especially the relationship between:

- Linear Regression
- Loss function
- Gradient
- `np.sum()` vs `np.mean()`
- Dataset size
- Feature scale
- Learning rate
- Gradient Descent divergence
- `inf`, `-inf`, and overflow
- Shape and broadcasting issues

The main lesson is:

> **Before choosing or implementing a loss function, understand its mathematical definition and how its scale affects the gradient and therefore the learning rate.**

---

# 1. Linear Regression Model

For one feature:

ŷᵢ = mxᵢ + b

Where:

- xᵢ = input feature
- yᵢ = actual target
- ŷᵢ = predicted value
- m = slope / weight
- b = intercept / bias

For multiple features, this extends to:

ŷ = Xw + b

---

# 2. Error / Residual

For each observation:

eᵢ = yᵢ - ŷᵢ

Since:

ŷᵢ = mxᵢ + b

we get:

eᵢ = yᵢ - (mxᵢ + b)

or:

eᵢ = yᵢ - mxᵢ - b

---

# 3. Two Important Loss Functions

There are two closely related formulations.

## A. Sum of Squared Errors (SSE)

SSE is:

J_SSE(m,b) = Σ [yᵢ - (mxᵢ + b)]²

In Python, the aggregation is conceptually:

np.sum(...)

SSE adds the squared errors of all observations.

---

## B. Mean Squared Error (MSE)

MSE is:

J_MSE(m,b) = (1/n) Σ [yᵢ - (mxᵢ + b)]²

In Python, the aggregation is conceptually:

np.mean(...)

MSE calculates the average squared error.

---

# 4. Important: Both SSE and MSE Are Correct

It is NOT correct to say:

> SSE is wrong and MSE is correct.

Both are mathematically valid.

The relationship is:

J_MSE = (1/n) J_SSE

Therefore, they have the same optimal values of m and b.

The important difference is the **scale of the gradient**.

---

# 5. Deriving the Gradient for SSE

Start with:

J_SSE = Σ [yᵢ - mxᵢ - b]²

## Derivative with respect to b

∂J/∂b = -2 Σ [yᵢ - mxᵢ - b]

This corresponds to:

-2 * np.sum(y - mX - b)

So the original `np.sum()` approach was mathematically correct for SSE.

---

## Derivative with respect to m

∂J/∂m = -2 Σ [(yᵢ - mxᵢ - b)xᵢ]

This corresponds to:

-2 * np.sum((y - mX - b) * X)

Again, this is mathematically correct for SSE.

---

# 6. Deriving the Gradient for MSE

Start with:

J_MSE = (1/n) Σ [yᵢ - mxᵢ - b]²

## Derivative with respect to b

∂J/∂b = -(2/n) Σ [yᵢ - mxᵢ - b]

This can be written as:

-2 * mean(y - mX - b)

---

## Derivative with respect to m

∂J/∂m = -(2/n) Σ [(yᵢ - mxᵢ - b)xᵢ]

This can be written as:

-2 * mean((y - mX - b) * X)

---

# 7. Relationship Between SSE and MSE Gradients

The most important relationship is:

∇J_MSE = (1/n) ∇J_SSE

Therefore:

∇J_SSE = n ∇J_MSE

This means that if there are 95,232 observations:

∇J_SSE ≈ 95,232 × ∇J_MSE

for the same parameters and data.

So the SSE gradient can be roughly 95,232 times larger than the MSE gradient.

---

# 8. Why Did Gradient Descent Diverge?

The dataset contained approximately:

n = 95,232

observations.

The original implementation used:

np.sum()

Therefore, the gradient was accumulated across all 95,232 observations.

This can make the gradient very large.

Gradient Descent updates parameters using:

θ_new = θ_old - α∇J(θ)

For the slope:

m_new = m_old - α(∂J/∂m)

For the intercept:

b_new = b_old - α(∂J/∂b)

If the gradient is extremely large, the parameter update can also be extremely large.

This can create a feedback loop:

Large gradient
    ↓
Large parameter update
    ↓
m and/or b become large
    ↓
Predictions become large
    ↓
Errors become large
    ↓
Gradient becomes even larger
    ↓
Even larger update
    ↓
Divergence
    ↓
inf / -inf / overflow

---

# 9. What Happened in This Dataset?

The learned parameters eventually became approximately:

m = 2.02 × 10^161

b = 6.77 × 10^159

These values are clearly not reasonable for this regression problem.

They indicate that Gradient Descent was diverging rather than converging.

Eventually, predictions became so large that floating-point arithmetic overflowed.

This resulted in:

inf

and later:

-inf

when calculating R².

---

# 10. Why Did `r2_score` Produce `-inf`?

The R² calculation contains a squared error term:

(y_true - y_pred)²

If y_pred becomes extremely large, this operation can exceed the numerical range of floating-point numbers.

NumPy then reports:

RuntimeWarning: overflow encountered in square

and values may become:

inf

The R² calculation then becomes invalid and may result in:

-inf

Important:

> `-inf` from R² was NOT the original problem.

The real problem occurred earlier when Gradient Descent caused m and b to explode.

---

# 11. Dataset Size vs Feature Magnitude

There are two separate factors that can make gradients large.

## A. Number of observations

With SSE:

∂J/∂m = -2 Σ(eᵢxᵢ)

Every observation contributes to the sum.

Therefore, increasing the number of observations can increase the magnitude of the raw gradient.

With MSE:

∂J/∂m = -(2/n) Σ(eᵢxᵢ)

The division by n normalizes the gradient.

---

## B. Feature magnitude

The slope gradient contains:

eᵢxᵢ

Therefore, large values of X can also create large gradient contributions.

For example:

Error = 100
X = 9000

Then:

Error × X = 900,000

That is already a large contribution before summing over the dataset.

---

# 12. Important Correction About Squaring

It is tempting to say:

> "The gradient became huge because the errors were squared."

This is only partly correct.

The loss contains the squared error:

e²

But after differentiation, the slope gradient contains:

-2ex

because:

d(e²)/dm = 2e(de/dm)

and:

de/dm = -x

Therefore:

d(e²)/dm = -2ex

So for the slope, the important term is:

e × x

not simply:

e².

The squared error determines the loss, while differentiation produces the gradient.

---

# 13. Why MSE Is Often More Convenient

MSE is often convenient for Gradient Descent because:

## 1. Its scale is less dependent on dataset size

MSE averages the error:

MSE = (1/n) Σe²

instead of accumulating it:

SSE = Σe²

---

## 2. Its gradient is easier to control

∇MSE = (1/n)∇SSE

This generally makes learning-rate selection easier.

---

## 3. Easier comparison across datasets

Because MSE is an average, its scale is less directly affected by how many observations are present.

---

# 14. Can SSE Still Be Used?

YES.

You can perform Gradient Descent using SSE.

Your original mathematics:

-2 * np.sum(...)

is valid.

However, because the gradient is larger, the learning rate may need to be much smaller.

Since:

∇SSE = n∇MSE

the learning rate for SSE may need to be roughly n times smaller to produce similarly sized parameter updates.

This is only a rough relationship.

The exact stable learning rate also depends on:

- feature scale
- target scale
- data distribution
- initialization
- optimization setup

---

# 15. Learning Rate Is Not Universal

There is no single universally correct learning rate.

The appropriate learning rate depends on:

- Loss-function scaling
- Feature scale
- Target scale
- Number of samples
- Model structure
- Optimization algorithm

Therefore:

> Changing the loss-function normalization can require changing the learning rate.

Similarly:

> Changing feature scaling can require changing the learning rate.

---

# 16. Feature Scaling

If features have very large values, scaling can make Gradient Descent easier.

For example:

Original:

10000
20000
30000
40000

After standardization, values may become approximately:

-1.3
-0.4
0.4
1.3

Now the term:

e × x

is much easier to control numerically.

Feature scaling is especially important for Gradient Descent-based algorithms.

---

# 17. Shape and Broadcasting Problem

Another issue encountered during implementation was the difference between:

X.shape = (n, 1)

and:

y.shape = (n,)

For scikit-learn, a single feature should generally be 2D:

X → (n, 1)

For example:

[[25],
 [26],
 [27]]

The target can be:

y → (n,)

For manual NumPy calculations, converting X to:

X → (n,)

can make element-wise calculations easier.

---

# 18. Why Did the 67.6 GB MemoryError Occur?

If Pandas/NumPy receives:

X → (n, 1)

and:

y → (n,)

and the operations are performed incorrectly, broadcasting/alignment can produce:

(n, n)

instead of:

(n,)

With:

n = 95,232

this becomes:

95,232 × 95,232

which is approximately 9 billion values.

At float64 precision, this can require around:

67.6 GiB

of memory.

This produced:

MemoryError:
Unable to allocate 67.6 GiB

Therefore:

> Always check the shapes of X and y before vectorized calculations.

---

# 19. Useful Shape Checks

Check:

X_train.shape
y_train.shape

For one feature:

X_train → (n, 1)
y_train → (n,)

For manual calculations, you can convert:

X = X.to_numpy().ravel()
y = y.to_numpy().ravel()

giving:

X → (n,)
y → (n,)

Now operations are element-wise:

y - mX - b

and produce:

(n,)

instead of accidentally producing:

(n, n)

---

# 20. Gradient Descent Workflow

The complete process is:

1. Initialize parameters
2. Make predictions
3. Calculate errors
4. Calculate loss
5. Calculate gradients
6. Update parameters
7. Repeat

Mathematically:

θ_new = θ_old - α∇J(θ)

For linear regression:

m_new = m_old - α(∂J/∂m)

b_new = b_old - α(∂J/∂b)

---

# 21. Before Choosing or Implementing a Loss Function

This is the main checklist to remember.

## Step 1 — Define the loss mathematically

Do not immediately write:

np.sum()

or:

np.mean()

First determine:

What exactly am I minimizing?

For example:

SSE = Σe²

or:

MSE = (1/n)Σe²

---

## Step 2 — Derive the gradient

Differentiate the actual loss you chose.

If the loss contains:

1/n

then its gradient also contains:

1/n.

Do not mix the loss definition and gradient from different formulations.

---

## Step 3 — Check feature scale

Ask:

How large are my X values?

Large X values can produce large gradients.

---

## Step 4 — Check dataset size

Ask:

How many samples do I have?

If using a sum-based loss, remember:

gradient magnitude can scale with n.

---

## Step 5 — Choose the learning rate accordingly

The learning rate must be considered together with:

- loss scaling
- feature scale
- target scale
- dataset size

---

## Step 6 — Check dimensions

For one feature:

X → (n, 1)
y → (n,)

For manual NumPy calculations:

X → (n,)
y → (n,)

Avoid accidental broadcasting.

---

## Step 7 — Monitor the parameters

Check:

m
b
loss

If they suddenly become:

1e10
1e50
1e100
...

Gradient Descent is probably diverging.

---

## Step 8 — Watch for numerical problems

If you see:

inf
-inf
nan
overflow

stop and investigate the optimization.

Do not simply continue to calculate evaluation metrics.

---

# 22. Core Mental Model

The most important relationship to remember is:

Loss definition
      ↓
Gradient
      ↓
Gradient magnitude
      ↓
Learning rate
      ↓
Parameter update
      ↓
Convergence or divergence

Therefore:

> The loss function and learning rate cannot be considered independently.

---

# 23. Final Takeaway

SSE:

J_SSE = Σe²

MSE:

J_MSE = (1/n)Σe²

Relationship:

J_MSE = (1/n)J_SSE

Gradients:

∇J_MSE = (1/n)∇J_SSE

Therefore:

∇J_SSE = n∇J_MSE

Both losses have the same optimal parameters.

The difference is mainly the scale of the objective and its gradient.

The key lesson is:

> **Before implementing a loss function, understand whether you are summing or averaging, derive its gradient, check the scale of your features and dataset, and then choose an appropriate learning rate.**

This becomes increasingly important when moving from Linear Regression to:

- Logistic Regression
- Neural Networks
- Deep Learning
- Mini-batch Gradient Descent
- SGD
- Adam
- Other optimization algorithms