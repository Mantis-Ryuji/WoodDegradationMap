# Mathematics of SNV and spectral perturbations

Using the definitions of [preprocessing](design/preprocessing.md) and the [model and perturbations](chemomae_positioning.md),
this document derives the mean and norm constraints for SNV, TGN, and Fractional Shift.
Finite inputs and nonzero normalization denominators are assumed. See the [numerical implementation notes](numerical_implementation_notes.md) for numerical handling in the implementation.

<a id="snv-geometry"></a>

## 1 SNV constraints and geometry

### 1.1 Mean and norm

Consider one valid pixel and omit its index $`p`$. Let $`C=256`$ be the number of channels after interpolation, $`j=0,\ldots,C-1`$ the channel index, $`\widetilde{\boldsymbol{R}}\in\mathbb{R}^{C}`$ the interpolated reflectance, and $`\widetilde{R}_j`$ its $`j`$th component. The within-pixel mean $`\mu`$ and sample standard deviation $`s>0`$ are the same quantities as the per-pixel $`\mu_p,s_p`$. Here, $`\boldsymbol{1}`$ is the $`C`$-dimensional all-ones vector, superscript $`\mathsf T`$ denotes transpose, and $`\|\cdot\|_2`$ is the L2 norm. For the SNV spectrum $`\boldsymbol{x}=(\widetilde{\boldsymbol{R}}-\mu\boldsymbol{1})/s`$,

```math
\boldsymbol{1}^{\mathsf T}\boldsymbol{x}
=
\frac{1}{s}
\left(\sum_{j=0}^{C-1}\widetilde{R}_j-C\mu\right)
=0
\qquad\text{(1)}
```

Also, since $`s^2=(C-1)^{-1}\sum_j(\widetilde{R}_j-\mu)^2`$,

```math
\|\boldsymbol{x}\|_2^2
=
\frac{\sum_j(\widetilde{R}_j-\mu)^2}{s^2}
=C-1.
\qquad\text{(2)}
```

Thus, writing the norm of an SNV spectrum as $`\rho=\sqrt{C-1}`$, the set $`\mathcal{S}_{\mathrm{SNV}}`$ satisfying SNV's mean and norm constraints is

```math
\mathcal{S}_{\mathrm{SNV}}
=
\left\{
\boldsymbol{x}\in\mathbb{R}^{C}:
\boldsymbol{1}^{\mathsf T}\boldsymbol{x}=0,\ 
\|\boldsymbol{x}\|_2=\rho
\right\}
\qquad\text{(3)}
```

SNV inputs belong to this set. The zero-mean hyperplane has dimension $`C-1`$, and the sphere of radius $`\rho`$ within it has dimension $`C-2`$. For $`C=256`$, $`\rho=\sqrt{255}`$ and the sphere has dimension 254. This is the set of points satisfying the SNV constraints, not the set of physically realizable material spectra itself.

### 1.2 Centering as a projection

Use the centering matrix $`P=I_C-\boldsymbol{1}\boldsymbol{1}^{\mathsf T}/C`$ to remove the mean, where $`I_C`$ is the $`C`$-dimensional identity matrix. It satisfies

```math
P^{\mathsf T}=P,\qquad
P^2=P,\qquad
P\boldsymbol{1}=\boldsymbol{0}
\qquad\text{(4)}
```

Here, $`\boldsymbol{0}`$ is the $`C`$-dimensional zero vector. Since the centered interpolated reflectance $`P\widetilde{\boldsymbol{R}}`$ has zero mean, SNV can be written as

```math
\boldsymbol{x}
=
\rho\frac{P\widetilde{\boldsymbol{R}}}{\|P\widetilde{\boldsymbol{R}}\|_2}
\qquad\text{(5)}
```

provided that $`P\widetilde{\boldsymbol{R}}\neq\boldsymbol{0}`$.

Applying a common positive scale $`\kappa>0`$ and offset $`\beta`$ across all bands before SNV gives
$`P(\kappa\widetilde{\boldsymbol{R}}+\beta\boldsymbol{1})=\kappa P\widetilde{\boldsymbol{R}}`$. Therefore,

```math
\mathop{\mathrm{SNV}}\nolimits(\kappa\widetilde{\boldsymbol{R}}+\beta\boldsymbol{1})
=
\mathop{\mathrm{SNV}}\nolimits(\widetilde{\boldsymbol{R}})
\qquad\text{(6)}
```

This property removes a common offset and scale within each pixel; it does not remove wavelength-dependent distortions or all measurement variation.

<a id="tgn-geometry"></a>

## 2 TGN tangent directions and constraint preservation

### 2.1 Projection onto the tangent space

Let $`C=256`$ be the number of channels and $`\mathcal{S}_{\mathrm{SNV}}`$ the SNV constraint set. For an SNV input $`\boldsymbol{x}\in\mathcal{S}_{\mathrm{SNV}}`$, let $`\rho=\sqrt{C-1}`$ be its norm and $`\boldsymbol{n}=\boldsymbol{x}/\rho`$ its unit direction. Tangent directions at $`\boldsymbol{x}`$ lie in the zero-mean hyperplane and are orthogonal to the radial direction $`\boldsymbol{n}`$. Denoting this set of directions, the tangent space, by $`\mathcal{T}_{\boldsymbol{n}}`$,

```math
\mathcal{T}_{\boldsymbol{n}}
=
\left\{
\boldsymbol{q}_{\mathrm{raw}}\in\mathbb{R}^{C}:
\boldsymbol{1}^{\mathsf T}\boldsymbol{q}_{\mathrm{raw}}=0,\ 
\boldsymbol{n}^{\mathsf T}\boldsymbol{q}_{\mathrm{raw}}=0
\right\}
\qquad\text{(7)}
```

The vector $`\boldsymbol{q}_{\mathrm{raw}}`$ in this set definition is an unnormalized tangent vector. Use a Gaussian candidate direction $`\boldsymbol{\epsilon}\sim\mathcal{N}(\boldsymbol{0},I_C)`$. Here, $`\mathcal{N}(\boldsymbol{0},I_C)`$ is the $`C`$-dimensional normal distribution with zero mean and identity covariance, and $`P`$ is the centering matrix in Eq. (4). Construct

```math
\boldsymbol{q}_{\mathrm{raw}}
=
P\boldsymbol{\epsilon}
-
\boldsymbol{n}
\left(\boldsymbol{n}^{\mathsf T}P\boldsymbol{\epsilon}\right)
\qquad\text{(8)}
```

Then $`\boldsymbol{1}^{\mathsf T}P=\boldsymbol{0}^{\mathsf T}`$ and
$`\boldsymbol{1}^{\mathsf T}\boldsymbol{n}=0`$ imply
$`\boldsymbol{1}^{\mathsf T}\boldsymbol{q}_{\mathrm{raw}}=0`$. Also,

```math
\boldsymbol{n}^{\mathsf T}\boldsymbol{q}_{\mathrm{raw}}
=
\boldsymbol{n}^{\mathsf T}P\boldsymbol{\epsilon}
-
(\boldsymbol{n}^{\mathsf T}\boldsymbol{n})
\boldsymbol{n}^{\mathsf T}P\boldsymbol{\epsilon}
=0
\qquad\text{(9)}
```

Thus, when $`\|\boldsymbol{q}_{\mathrm{raw}}\|_2>0`$,
$`\boldsymbol{q}=\boldsymbol{q}_{\mathrm{raw}}/\|\boldsymbol{q}_{\mathrm{raw}}\|_2`$ is a zero-mean unit tangent direction.

Since $`P\boldsymbol{n}=\boldsymbol{n}`$, Eq. (8) can also be written as
$`\boldsymbol{q}_{\mathrm{raw}}=(P-\boldsymbol{n}\boldsymbol{n}^{\mathsf T})\boldsymbol{\epsilon}`$.
The matrix $`P-\boldsymbol{n}\boldsymbol{n}^{\mathsf T}`$ is symmetric and idempotent,
and is the orthogonal projection onto $`\mathcal{T}_{\boldsymbol{n}}`$. The tangent space has dimension $`C-2`$.

### 2.2 Mean and norm after rotation

For rotation angle $`\alpha`$ in radians and unit tangent direction $`\boldsymbol{q}`$, define the TGN output as $`\boldsymbol{x}_{\mathrm{TGN}}=\rho(\cos\alpha\,\boldsymbol{n}+\sin\alpha\,\boldsymbol{q})`$.
Both directions have zero mean, so $`\boldsymbol{1}^{\mathsf T}\boldsymbol{x}_{\mathrm{TGN}}=0`$.
Since $`\boldsymbol{n}^{\mathsf T}\boldsymbol{q}=0`$,

```math
\begin{aligned}
\|\boldsymbol{x}_{\mathrm{TGN}}\|_2^2
&=
\rho^2\left(
\cos^2\alpha\,\|\boldsymbol{n}\|_2^2
+
\sin^2\alpha\,\|\boldsymbol{q}\|_2^2
+
2\cos\alpha\sin\alpha\,\boldsymbol{n}^{\mathsf T}\boldsymbol{q}
\right)\\
&=\rho^2
\end{aligned}
\qquad\text{(10)}
```

TGN is therefore a transformation within $`\mathcal{S}_{\mathrm{SNV}}`$.

### 2.3 Distribution of the rotation angle

The cosine similarity between input and output is

```math
\frac{\boldsymbol{x}^{\mathsf T}\boldsymbol{x}_{\mathrm{TGN}}}
{\|\boldsymbol{x}\|_2\|\boldsymbol{x}_{\mathrm{TGN}}\|_2}
=\cos\alpha
\qquad\text{(11)}
```

Within the adopted range $`0\leq\alpha\leq\pi/36`$, their angle is $`\alpha`$.

This study uses $`\alpha\sim\mathcal{U}(0,\pi/36)`$. Here, $`\mathcal{U}`$ denotes a continuous uniform distribution over the specified interval, and $`\pi`$ is pi; the upper limit corresponds to 5 degrees. This differs from sampling cosine similarity $`\cos\alpha`$ uniformly. Although candidate directions are Gaussian, the final residual after projection, unit normalization, and rotation cannot be treated as independent additive Gaussian noise.

<a id="fractional-shift"></a>

## 3 Fractional Shift interpolation and reprojection

### 3.1 Reference positions and endpoint-value extension

Let $`\boldsymbol{x}`$ be an SNV input with $`C=256`$ channels, $`j=0,\ldots,C-1`$ the output channel index, and $`\delta`$ the shift in channel units. For output position $`j`$, let $`\zeta_j`$ be the continuous reference position, $`\iota_j^-,\iota_j^+`$ its left and right integer indices, and $`\omega_j^{\mathrm{FS}}`$ the interpolation weight on the right value. Define

```math
\zeta_j=j-\delta,\qquad
\iota_j^-=\lfloor\zeta_j\rfloor,\qquad
\iota_j^+=\iota_j^-+1,\qquad
\omega_j^{\mathrm{FS}}=\zeta_j-\lfloor\zeta_j\rfloor
\qquad\text{(12)}
```

The floor function $`\lfloor\cdot\rfloor`$ returns the largest integer no greater than its argument. The index $`\iota_j^+`$ is always the left index plus one, including when the reference position is an integer. Clip the indices to the spectrum's endpoints, rather than image boundaries:

$`\bar{\iota}_j^-=\min(C-1,\max(0,\iota_j^-))`$,
$`\bar{\iota}_j^+=\min(C-1,\max(0,\iota_j^+))`$.

The superscripts $`-,+`$ distinguish left and right reference points; they are not numerical signs. The $`j`$th component of the linear interpolation operator $`S_\delta`$ is

```math
[S_\delta(\boldsymbol{x})]_j
=
(1-\omega_j^{\mathrm{FS}})x_{\bar{\iota}_j^-}
+\omega_j^{\mathrm{FS}}x_{\bar{\iota}_j^+}
\qquad\text{(13)}
```

Here, $`x_{\bar{\iota}_j^-},x_{\bar{\iota}_j^+}`$ are the input spectrum values at the reference indices. Outside the valid range, both indices map to the same endpoint, extending that endpoint's value. Positive $`\delta`$ shifts features toward larger indices, and negative $`\delta`$ toward smaller indices.

If $`\Delta\lambda`$ is the wavelength spacing between adjacent channels on the uniform grid, the interpolation's axial displacement is $`\delta\Delta\lambda`$. The complete change after recentering and renormalization is not identified with a simple physical absorption-band displacement or a measured calibration error.

### 3.2 Returning to the constraint set

The interpolation matrix is generally not orthogonal and includes endpoint handling, so interpolation alone need not preserve the input mean or norm. Let $`P`$ be the centering matrix in Eq. (4), and $`\rho=\sqrt{C-1}`$ the SNV input norm. Define the mean-centered interpolated spectrum as $`\boldsymbol{y}=P S_\delta(\boldsymbol{x})`$. For $`\|\boldsymbol{y}\|_2>0`$, define the final Fractional Shift output $`\boldsymbol{x}_{\mathrm{FS}}`$ as

```math
\boldsymbol{x}_{\mathrm{FS}}
=
\rho\frac{\boldsymbol{y}}{\|\boldsymbol{y}\|_2}
\qquad\text{(14)}
```

The properties of $`P`$ give $`\boldsymbol{1}^{\mathsf T}\boldsymbol{x}_{\mathrm{FS}}=0`$,
and normalization gives $`\|\boldsymbol{x}_{\mathrm{FS}}\|_2=\rho`$. This reprojection places the output in $`\mathcal{S}_{\mathrm{SNV}}`$.

Unlike TGN, a fixed $`\delta`$ does not uniquely determine the angle between input and output, because interpolation depends on the original spectral shape and endpoint values. Preservation of SNV constraints by reprojection is not proof of chemical information preservation.

---

## Definition and implementation references

The underlying definitions appear in the [model and perturbation description](chemomae_positioning.md) and [experiment protocol](design/experiment_protocol.md).
See ChemoMAE v0.2.2's `SpectraAugmenter` and the [calling code](../src/wood_degradation_map/experiments/neural.py) for the implementation.
