"""Literal Python translation of subsolv.m (Krister Svanberg, Dec 2006).

Solves the convex, separable MMA subproblem built by :func:`fdtdx.optimization.mmasub.mmasub`
via a primal-dual interior-point (Newton) method. Ported line-for-line from the original
MATLAB so that fdtdx does not depend on any third-party MMA implementation.
"""

import numpy as np
from loguru import logger


def subsolv(
    m: int,
    n: int,
    epsimin: float,
    low: np.ndarray,
    upp: np.ndarray,
    alfa: np.ndarray,
    beta: np.ndarray,
    p0: np.ndarray,
    q0: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    a0: float,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solves the MMA subproblem.

    minimize   SUM[ p0j/(uppj-xj) + q0j/(xj-lowj) ] + a0*z + SUM[ ci*yi + 0.5*di*(yi)^2 ],
    subject to SUM[ pij/(uppj-xj) + qij/(xj-lowj) ] - ai*z - yi <= bi,
               alfaj <= xj <= betaj,  yi >= 0,  z >= 0.

    Args:
        m (int): Number of general constraints.
        n (int): Number of variables x_j.
        epsimin (float): Convergence tolerance on the interior-point residual.
        low (np.ndarray): Lower asymptotes, shape (n, 1).
        upp (np.ndarray): Upper asymptotes, shape (n, 1).
        alfa (np.ndarray): Lower move-limited bounds on x, shape (n, 1).
        beta (np.ndarray): Upper move-limited bounds on x, shape (n, 1).
        p0 (np.ndarray): Objective approximation coefficients, shape (n, 1).
        q0 (np.ndarray): Objective approximation coefficients, shape (n, 1).
        P (np.ndarray): Constraint approximation coefficients, shape (m, n).
        Q (np.ndarray): Constraint approximation coefficients, shape (m, n).
        a0 (float): Constant a_0 in the term a_0*z.
        a (np.ndarray): Constants a_i in the terms a_i*z, shape (m, 1).
        b (np.ndarray): Constraint right-hand sides, shape (m, 1).
        c (np.ndarray): Constants c_i in the terms c_i*y_i, shape (m, 1).
        d (np.ndarray): Constants d_i in the terms 0.5*d_i*(y_i)^2, shape (m, 1).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray]: xmma, ymma, zmma, lamma, xsimma, etamma,
            mumma, zetmma, smma -- the optimal primal variables x_j (n, 1) and y_i, z
            (m, 1)/(1, 1), and the Lagrange multipliers/slacks of the subproblem.
    """
    een = np.ones((n, 1))
    eem = np.ones((m, 1))
    epsi = 1.0
    x = 0.5 * (alfa + beta)
    y = eem.copy()
    z = np.array([[1.0]])
    lam = eem.copy()
    xsi = np.maximum(een / (x - alfa), een)
    eta = np.maximum(een / (beta - x), een)
    mu = np.maximum(eem, 0.5 * c)
    zet = np.array([[1.0]])
    s = eem.copy()
    itera = 0
    while epsi > epsimin:
        epsvecn = epsi * een
        epsvecm = epsi * eem
        ux1 = upp - x
        xl1 = x - low
        ux2 = ux1 * ux1
        xl2 = xl1 * xl1
        uxinv1 = een / ux1
        xlinv1 = een / xl1
        plam = p0 + P.T @ lam
        qlam = q0 + Q.T @ lam
        gvec = P @ uxinv1 + Q @ xlinv1
        dpsidx = plam / ux2 - qlam / xl2
        rex = dpsidx - xsi + eta
        rey = c + d * y - mu - lam
        rez = a0 - zet - a.T @ lam
        relam = gvec - a * z - y + s - b
        rexsi = xsi * (x - alfa) - epsvecn
        reeta = eta * (beta - x) - epsvecn
        remu = mu * y - epsvecm
        rezet = zet * z - epsi
        res = lam * s - epsvecm
        residu1 = np.concatenate((rex, rey, rez), axis=0)
        residu2 = np.concatenate((relam, rexsi, reeta, remu, rezet, res), axis=0)
        residu = np.concatenate((residu1, residu2), axis=0)
        residunorm = float(np.sqrt((residu * residu).sum()))
        residumax = float(np.max(np.abs(residu)))
        ittt = 0
        while residumax > 0.9 * epsi and ittt < 200:
            ittt += 1
            itera += 1
            ux1 = upp - x
            xl1 = x - low
            ux2 = ux1 * ux1
            xl2 = xl1 * xl1
            ux3 = ux1 * ux2
            xl3 = xl1 * xl2
            uxinv1 = een / ux1
            xlinv1 = een / xl1
            uxinv2 = een / ux2
            xlinv2 = een / xl2
            plam = p0 + P.T @ lam
            qlam = q0 + Q.T @ lam
            gvec = P @ uxinv1 + Q @ xlinv1
            # GG = P*spdiags(uxinv2,0,n,n) - Q*spdiags(xlinv2,0,n,n); a diagonal matrix on
            # the right scales columns, which is the same as broadcasting the diagonal as a
            # row vector -- avoids materializing an (n, n) matrix.
            GG = P * uxinv2.T - Q * xlinv2.T
            dpsidx = plam / ux2 - qlam / xl2
            delx = dpsidx - epsvecn / (x - alfa) + epsvecn / (beta - x)
            dely = c + d * y - lam - epsvecm / y
            delz = a0 - a.T @ lam - epsi / z
            dellam = gvec - a * z - y - b + epsvecm / lam
            diagx = plam / ux3 + qlam / xl3
            diagx = 2 * diagx + xsi / (x - alfa) + eta / (beta - x)
            diagxinv = een / diagx
            diagy = d + mu / y
            diagyinv = eem / diagy
            diaglam = s / lam
            diaglamyi = diaglam + diagyinv
            if m < n:
                blam = dellam + dely / diagy - GG @ (delx / diagx)
                bb = np.concatenate((blam, delz), axis=0)
                Alam = np.diagflat(diaglamyi) + (GG * diagxinv.T) @ GG.T
                AA = np.concatenate(
                    (
                        np.concatenate((Alam, a), axis=1),
                        np.concatenate((a.T, -zet / z), axis=1),
                    ),
                    axis=0,
                )
                solut = np.linalg.solve(AA, bb)
                dlam = solut[0:m]
                dz = solut[m : m + 1]
                dx = -delx / diagx - (GG.T @ dlam) / diagx
            else:
                diaglamyiinv = eem / diaglamyi
                dellamyi = dellam + dely / diagy
                Axx = np.diagflat(diagx) + (GG.T * diaglamyiinv.T) @ GG
                azz = zet / z + a.T @ (a / diaglamyi)
                axz = -GG.T @ (a / diaglamyi)
                bx = delx + GG.T @ (dellamyi / diaglamyi)
                bz = delz - a.T @ (dellamyi / diaglamyi)
                AA = np.concatenate(
                    (
                        np.concatenate((Axx, axz), axis=1),
                        np.concatenate((axz.T, azz), axis=1),
                    ),
                    axis=0,
                )
                bb = np.concatenate((-bx, -bz), axis=0)
                solut = np.linalg.solve(AA, bb)
                dx = solut[0:n]
                dz = solut[n : n + 1]
                dlam = (GG @ dx) / diaglamyi - dz * (a / diaglamyi) + dellamyi / diaglamyi
            dy = -dely / diagy + dlam / diagy
            dxsi = -xsi + epsvecn / (x - alfa) - (xsi * dx) / (x - alfa)
            deta = -eta + epsvecn / (beta - x) + (eta * dx) / (beta - x)
            dmu = -mu + epsvecm / y - (mu * dy) / y
            dzet = -zet + epsi / z - zet * dz / z
            ds = -s + epsvecm / lam - (s * dlam) / lam
            xx = np.concatenate((y, z, lam, xsi, eta, mu, zet, s), axis=0)
            dxx = np.concatenate((dy, dz, dlam, dxsi, deta, dmu, dzet, ds), axis=0)

            stepxx = -1.01 * dxx / xx
            stmxx = float(np.max(stepxx))
            stepalfa = -1.01 * dx / (x - alfa)
            stmalfa = float(np.max(stepalfa))
            stepbeta = 1.01 * dx / (beta - x)
            stmbeta = float(np.max(stepbeta))
            stmalbe = max(stmalfa, stmbeta)
            stmalbexx = max(stmalbe, stmxx)
            stminv = max(stmalbexx, 1.0)
            steg = 1.0 / stminv

            xold = x.copy()
            yold = y.copy()
            zold = z.copy()
            lamold = lam.copy()
            xsiold = xsi.copy()
            etaold = eta.copy()
            muold = mu.copy()
            zetold = zet.copy()
            sold = s.copy()

            itto = 0
            resinew = 2 * residunorm
            while resinew > residunorm and itto < 50:
                itto += 1
                x = xold + steg * dx
                y = yold + steg * dy
                z = zold + steg * dz
                lam = lamold + steg * dlam
                xsi = xsiold + steg * dxsi
                eta = etaold + steg * deta
                mu = muold + steg * dmu
                zet = zetold + steg * dzet
                s = sold + steg * ds
                ux1 = upp - x
                xl1 = x - low
                ux2 = ux1 * ux1
                xl2 = xl1 * xl1
                uxinv1 = een / ux1
                xlinv1 = een / xl1
                plam = p0 + P.T @ lam
                qlam = q0 + Q.T @ lam
                gvec = P @ uxinv1 + Q @ xlinv1
                dpsidx = plam / ux2 - qlam / xl2
                rex = dpsidx - xsi + eta
                rey = c + d * y - mu - lam
                rez = a0 - zet - a.T @ lam
                relam = gvec - a * z - y + s - b
                rexsi = xsi * (x - alfa) - epsvecn
                reeta = eta * (beta - x) - epsvecn
                remu = mu * y - epsvecm
                rezet = zet * z - epsi
                res = lam * s - epsvecm
                residu1 = np.concatenate((rex, rey, rez), axis=0)
                residu2 = np.concatenate((relam, rexsi, reeta, remu, rezet, res), axis=0)
                residu = np.concatenate((residu1, residu2), axis=0)
                resinew = float(np.sqrt((residu * residu).sum()))
                steg = steg / 2.0
            residunorm = resinew
            residumax = float(np.max(np.abs(residu)))
            steg = 2.0 * steg
        if ittt > 198:
            logger.warning(f"subsolv: inner Newton loop did not converge (epsi={epsi}, ittt={ittt})")
        epsi = 0.1 * epsi
    xmma = x
    ymma = y
    zmma = z
    lamma = lam
    xsimma = xsi
    etamma = eta
    mumma = mu
    zetmma = zet
    smma = s
    return xmma, ymma, zmma, lamma, xsimma, etamma, mumma, zetmma, smma
