import numpy as np
import theano
import theano.tensor as tt

from pymc3.distributions import Continuous

solve_l = tt.slinalg.solve_lower_triangular
solve_u = tt.slinalg.solve_upper_triangular

__all__ = ['KalmanTheano', 'KalmanFilter']


class DimensionalityError(Exception):
    pass


def _filter(y, Phi, Q, L, c, H, Sv, d, s, P):
    """
    Perform 1 filtering step.  The previous state estimates and log likelihood
    up to the previous time step being given by (s, P).  The rest of
    the arguments are parameters for the state space model.
    """
    s_fwd, P_fwd, y_est, y_est_var = _predict(s, P, Phi, Q, L, c, H, Sv, d)

    # Cholesky factor and estimation error
    Ly_est_var = tt.slinalg.cholesky(y_est_var)
    err = y - y_est

    # make corrections
    s_cor, P_cor = _correct(s_fwd, Ly_est_var, err, P_fwd, Phi, H)

    # Accumulate loglikelihood
    log_l = _log_likelihood(err, Ly_est_var)
    return s_cor, P_cor, log_l


def _predict(s, P, Phi, Q, L, c, H, Sv, d):
    """
    Kalman filter prediction step
    """
    # State propogation
    s_fwd = tt.dot(Phi, s) + c
    P_fwd = tt.dot(tt.dot(Phi, P), Phi.T) + tt.dot(tt.dot(L, Q), L.T)

    # Output estimate and uncertainty
    y_est = tt.dot(H, s_fwd) + d
    y_est_var = tt.dot(tt.dot(H, P_fwd), H.T) + Sv
    return s_fwd, P_fwd, y_est, y_est_var


def _correct(s_fwd, Ly, err, P_fwd, Phi, H):
    K = tt.dot(P_fwd, solve_u(Ly.T, solve_l(Ly, H)).T)
    s_cor = s_fwd + tt.dot(K, err)
    KL = tt.dot(K, Ly)
    P_cor = P_fwd - tt.dot(KL, KL.T)
    return s_cor, P_cor


def _log_likelihood(err, Ly):
    n = err.shape[0]  # Number of dimensions

    logdet = tt.log(tt.diag(Ly)).sum()
    vTSv = tt.nlinalg.norm(solve_l(Ly, err), 2)**2
    return -0.5 * (n * np.log(2 * np.pi) + logdet + vTSv)


class KalmanTheano(object):
    def __init__(self, Phi, Q, L, c, H, Sv, d, s0, P0, n, m, g):
        # NOTE: If identical matrices happen to be passed in, theano
        # NOTE: will recognize this can use references.  This can be
        # NOTE: confusing as the names given below need not "stick".

        # State transition
        self.Phi = tt.as_tensor_variable(Phi, name="Phi")

        # State innovations
        self.Q = tt.as_tensor_variable(Q, name="Q")

        # Innovations modifier
        self.L = tt.as_tensor_variable(L, name="L")

        # State structural component
        self.c = tt.as_tensor_variable(c, name="c")

        # Observation matrix
        self.H = tt.as_tensor_variable(H, name="H")

        # Observation noise variance
        self.Sv = tt.as_tensor_variable(Sv, name="Sv")

        # Observation structural component
        self.d = tt.as_tensor_variable(d, name="d")

        # Initial state mean
        self.s0 = tt.as_tensor_variable(s0, name="s0")

        # Initial state variance
        self.P0 = tt.as_tensor_variable(P0, name="P0")

        self.n = n  # Output dimension
        self.m = m  # State dimension
        self.g = g  # Innovations dimension (often m == g)

        self.tensors = [self.Phi, self.Q, self.L, self.c,
                        self.H, self.Sv, self.d]
        self.tensor_names = ["Phi", "Q", "L", "c",
                             "H", "Sv", "d"]
        self.tensor_dims = [2, 2, 2, 1, 2, 2, 1]  # Matrix or vector

        self._validate()
        return

    def _validate(self):
        sequences = []
        non_sequences = []

        def is_seq(tnsr, dim=1):
            ndim = tnsr.ndim
            if ndim == dim:
                return False
            elif ndim == dim + 1:
                return True
            else:
                raise DimensionalityError(
                    "Variable {} has {} dimensions, but "
                    "should have only {} or {}"
                    "".format(tnsr.name, ndim, dim, dim + 1))

        def append_seq(name, tnsr, expected_dim=1):
            if is_seq(tnsr, dim):
                sequences.append((tnsr, name))
            else:
                non_sequences.append((tnsr, name))

        for name, tnsr, dim in zip(self.tensor_names,
                                   self.tensors,
                                   self.tensor_dims):
            append_seq(name, tnsr, dim)

        self.sequences = sequences
        self.non_sequences = non_sequences
        return

    def filter(self, Y, **th_scan_kwargs):
        # Create function with correct ordering for scan
        fn = eval(
            "lambda {}: _filter(y, Phi, Q, L, c, H, Sv, d, s, P)"
            "".format(",".join(
                ["y"] +
                [tnsr_name[1] for tnsr_name in self.sequences] +
                ["s", "P"] +
                [tnsr_name[1] for tnsr_name in self.non_sequences])))

        (st, Pt, log_l), updates = theano.scan(
            fn=fn,
            sequences=[Y] + [tnsr_name[0] for tnsr_name in self.sequences],
            outputs_info=[dict(initial=self.s0),
                          dict(initial=self.P0),
                          None],
            non_sequences=[tnsr_name[0] for tnsr_name in self.non_sequences],
            strict=True,
            **th_scan_kwargs)
        return (st, Pt, log_l.sum()), updates


class KalmanFilter(Continuous):
    """
    Implements a generic Kalman filter in general state space form.

    Shape of the input tensors is given as a function of:

    * T: number of time steps,
    * n: size of the observation vector
    * m: size of the state vector
    * g: size of the disturbance vector in the transition equation

    The following rules define tensor dimension reductions allowed:

    * If a tensor is time-invariant, the time dimension T can be omitted
    * If n=1, all dimensions of size n can be omitted
    * If m=1 and g=1, all dimensions of size m and g can be omitted

    Parameters
    ----------
    Phi : tensor or numpy array, dimensions T x m x m
          Tensor relating the state vectors at times t - 1, t
    c : tensor or numpy array, dimensions T x m
        offset in the state transition equation
    Q : tensor or numpy array, dimensions T x g x g
        Covariance matrix of the disturbances in the transition equation
    L : tensor or numpy array, dimensions T x m x g
        Tensor applying transition equation disturbances to state space
    H : tensor or numpy array, dimensions T x n x m
          Tensor relating observation and state vectors
    d : tensor or numpy array, dimensions T x n
        Shift in the observation equation
    Sv : tensor or numpy array, dimensions T x n x n
         Covariance for the observation noise
    s0 : tensor or numpy array, dimensions n
         Mean of the initial state vector
    P0 : tensor or numpy array, dimensions n x n
         Covariance of the initial state vector
    *args, **kwargs
        Extra arguments passed to :class:`Continuous` initialization

    Notes
    -----

    The general state space form (SSF) applies to a multivariate time series,
    y(t), containing n elements. We suppose that there is some underlying
    or background "state" s(t) containing m elements:

    .. math :

        s(t) = Phi(t) s(t-1) + c(t) + L(t) \\w(t)\\,\\qquad
             \\w(t) \\sim \\mathcal{N}_g(0, Q(t))\\
        s(0) \\sim \\mathcal{N}_m(s0, P0)

    These state variables generate the data via the "observation" equations:

    .. math :

        y(t) = H(t) s(t) + d(t) + \\v(t)\\,\\qquad
             \\v(t) \\sim \\mathcal{N}_n(0, Sv(t))\\ ,

    Although s(t) is typically not observable, its dynamics are governed by a
    first-order Gauss-Markov process.  The entire model is amenable to
    exact inference.

    The matrix L (which would correspond to a cholesky factor of the state
    variance if Q = I) can be used to linearly transform the state innovations
    w(t) and can be useful for modelling low-rank innovations.
    """
    def __init__(self, Phi, Q, L, c, H, Sv, d, s0, P0, n, m, g,
                 *args, **kwargs):
        Continuous.__init__(self, *args, **kwargs)

        self._kalman_theano = KalmanTheano(Phi, Q, L, c, H, Sv, d, s0, P0,
                                           n, m, g, **kwargs)
        self.mean = tt.as_tensor_variable(0.)
        return

    def logp(self, Y):
        (_, _, log_p), _ = self._kalman_theano.filter(Y)
        return log_p


if __name__ == "NONAME":
    n = 3
    m = 10

    T = 2048
    phi = 0.99
    v = np.random.normal(size=(T, m))
    Y = np.zeros((T, m))
    Y[0, :] = v[0, :]
    for t, vt in enumerate(v[1:]):
        Y[t + 1, :] = phi * Y[t, :] + vt

    sv_tnsr = tt.vector("sv")
    Sv_tnsr = tt.diag(sv_tnsr)

    # def __init__(self, Phi, Q, L, c, H, Sv, d, s0, P0, n, m, g,
    K = KalmanTheano(Phi=0.92 * np.eye(n), Q=0.2 * np.eye(n),
                     L=np.eye(n), c=np.zeros(n),
                     H=np.random.normal(size=(m, n)),
                     Sv=Sv_tnsr,
                     d=np.zeros(m),
                     s0=np.zeros(n),
                     P0=10 * np.eye(n),
                     n=n, m=m, g=n)
    Y_tensor = tt.matrix("Y")
    (s, P, ll), _ = K.filter(Y_tensor)
    kf = theano.function(inputs=[Y_tensor, sv_tnsr], outputs=[s, P, ll],
                         mode=theano.Mode(optimizer="unsafe"))

    s, P, ll = kf(Y, 2 * np.ones(m))

    import pymc3 as pm

    with pm.Model() as model:
        # Phi, Q, L, c, H, Sv, d, s0, P0, n, m, g

        phi = pm.Normal("phi", shape=(1, 1))
        q = pm.HalfStudentT("q", nu=1.0, sd=2.0, shape=(1, 1))
        K = KalmanFilter("kf", phi, q,
                         np.array([[1.]]),
                         np.array([0.]),
                         np.array([[1.]]),
                         np.array([[0.0]]),
                         np.array([0.]),
                         np.array([0.]),
                         np.array([[10.]]),
                         1, 1, 1,
                         observed=y)

    with model:
        # approx = pm.fit(n=100, method="advi")
        trace = pm.sample_approx(approx, draws=500)
