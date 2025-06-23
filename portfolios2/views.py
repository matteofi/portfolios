import yfinance as yf
import pandas as pd
import numpy as np
import cvxpy as cp
from datetime import datetime
from scipy.optimize import minimize
from django.shortcuts import render


def gmvp(cov_matrix, short_selling):
    ones = np.ones(cov_matrix.shape[0])
    inv_cov = np.linalg.inv(cov_matrix)

    if short_selling:
        w = inv_cov @ ones / (ones.T @ inv_cov @ ones)
    else:
        n = cov_matrix.shape[0]
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, cov_matrix))
        constraints = [
            cp.sum(w) == 1,
            w >= 0
        ]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        w = np.array(w.value)

    return w

def tangency_portfolio(expected_returns, cov_matrix, short_selling=True, risk_free_rate=0.0, penalty=0.05):
    if isinstance(expected_returns, pd.Series):
        expected_returns = expected_returns.values
    expected_returns = np.array(expected_returns)
    
    n = len(expected_returns)
    excess_returns = expected_returns - risk_free_rate

    if short_selling:
        inv_cov = np.linalg.inv(cov_matrix)
        w = (inv_cov @ excess_returns) / (np.ones(n) @ inv_cov @ excess_returns)
    else:
        def negative_sharpe_penalized(w):
            port_return = np.dot(w, excess_returns)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            sharpe = port_return / port_vol if port_vol != 0 else 0
            penalty_term = penalty * np.sum(w ** 2)  # L2 penalty to discourage extreme weights
            return -sharpe + penalty_term

        # constraint: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        # bounds: long-only (0 <= w_i <= 1)
        bounds = [(0, 1) for _ in range(n)]
        w0 = np.ones(n) / n  # initial equal weights

        result = minimize(
            negative_sharpe_penalized,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-6}
        )

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        w = result.x

    return w

def efficient_frontier(cov_matrix, expected_returns, risk_free_rate=0, num_points=1000):
    results = []
    ones = np.ones(len(expected_returns))
    inv_cov = np.linalg.inv(cov_matrix)

    # compute constants A, B, C, D
    A = ones.T @ inv_cov @ expected_returns
    B = expected_returns.T @ inv_cov @ expected_returns
    C = ones.T @ inv_cov @ ones
    D = B * C - A ** 2

    # GMVP expected return
    mu_gmvp = A / C

    # Tangent portfolio return
    excess_returns = expected_returns - risk_free_rate
    weights_tp = inv_cov @ excess_returns
    weights_tp /= ones.T @ weights_tp

    mu_tp = weights_tp @ expected_returns  # tangent portfolio return

    # extend return range symmetrically around GMVP
    max_distance = max(
        abs(mu_tp - mu_gmvp),
        abs(expected_returns.min() - mu_gmvp),
        abs(expected_returns.max() - mu_gmvp)
    )

    min_ep = mu_gmvp - max_distance
    max_ep = mu_tp + max_distance

    ep_values = np.linspace(min_ep, max_ep, num_points)

    for ep in ep_values:
        # compute portfolio variance via efficient frontier formula
        var_p = (C * ep**2 - 2 * A * ep + B) / D
        if var_p < 0:
            continue  # skip invalid portfolios
        std_dev = np.sqrt(var_p)
        sharpe_ratio = (ep - risk_free_rate) / std_dev
        results.append({
            'rischio': std_dev,
            'rendimento': ep,
            'sharpe': sharpe_ratio
        })

    return results

def portfolio_view(request):
    context = {}
    if request.method == 'POST':
        start_date = request.POST.get('start_date')
        end_date = request.POST.get('end_date')

        # Validate dates
        try:
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()

            if end_date_obj < start_date_obj:
                context['error'] = 'Error: end date must be after start date.'
                return render(request, 'portfolios2/portfolios.html', context)
            if start_date_obj < datetime(1960, 1, 1).date():
                context['error'] = 'Error: start date cannot be before 01/01/1960.'
                return render(request, 'portfolios2/portfolios.html', context)
        except (ValueError, TypeError):
            context['error'] = 'Error: invalid dates provided.'
            return render(request, 'portfolios2/portfolios.html', context)

        tickers_input = request.POST.get('tickers', '')
        tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

        if len(tickers) < 2:
            context['error'] = 'Error: please enter at least two valid tickers, separated by commas.'
            return render(request, 'portfolios2/portfolios.html', context)

        short_selling = 'short_selling' in request.POST
        risk_free_choice = request.POST.get('risk_free_choice')
        custom_risk_free = request.POST.get('custom_risk_free')

        # Handle risk-free rate selection
        if risk_free_choice == '0':
            risk_free_rate = 0.0
        elif risk_free_choice == 'custom':
            try:
                risk_free_rate = float(custom_risk_free)
            except (ValueError, TypeError):
                context['error'] = 'Error: invalid custom risk-free rate (use a numeric value).'
                return render(request, 'portfolios2/portfolios.html', context)
        elif risk_free_choice == 'bil':
            try:
                bil_data = yf.download("BIL", start=start_date, end=end_date)['Close'].dropna()
                bil_returns = bil_data.pct_change().dropna()
                if bil_data.empty:
                    raise ValueError
                risk_free_rate = (bil_returns.mean() * 252).iloc[0]
            except Exception:
                context['error'] = 'Error retrieving BIL risk-free proxy.'
                return render(request, 'portfolios2/portfolios.html', context)
        else:
            risk_free_rate = 0.0

        # Download and validate price data
        try:
            raw_data = yf.download(tickers, start=start_date, end=end_date)
            data = raw_data.get('Close')

            if data is None or data.empty:
                raise ValueError('No closing price data.')

            found_tickers = data.columns.levels[1] if isinstance(data.columns, pd.MultiIndex) else data.columns
            missing_tickers = [t for t in tickers if t not in found_tickers]

            if missing_tickers or data.empty:
                raise ValueError('Error in tickers.')

            data = data.dropna()
            if data.empty:
                raise ValueError('Error in tickers.')
        except Exception:
            context['error'] = 'Error in tickers.'
            return render(request, 'portfolios2/portfolios.html', context)

        # Compute log returns and annualised moments
        log_returns = np.log(data / data.shift(1)).dropna()
        cov_matrix = log_returns.cov() * 252
        expected_returns = log_returns.mean() * 252

        # Compute GMVP stats
        gmvp_weights = gmvp(cov_matrix, short_selling)
        gmvp_exp = expected_returns @ gmvp_weights
        gmvp_var = gmvp_weights.T @ cov_matrix @ gmvp_weights
        gmvp_std = np.sqrt(gmvp_var)

        # Compute Tangent Portfolio stats
        tp_weights = tangency_portfolio(
            expected_returns, cov_matrix,
            short_selling=short_selling,
            risk_free_rate=risk_free_rate
        )
        tp_exp = expected_returns @ tp_weights
        tp_var = tp_weights.T @ cov_matrix @ tp_weights
        tp_std = np.sqrt(tp_var)

        # Efficient frontier data
        frontier_data = efficient_frontier(
            cov_matrix, expected_returns,
            risk_free_rate=risk_free_rate
        )

        # Sharpe ratios
        gmvp_sharpe = (gmvp_exp - risk_free_rate) / gmvp_std
        tp_sharpe = (tp_exp - risk_free_rate) / tp_std

        # Cumulative performance series
        gmvp_cumulative = (1 + (log_returns @ gmvp_weights)).cumprod()
        tp_cumulative = (1 + (log_returns @ tp_weights)).cumprod()

        context = {
            'tickers': tickers,
            'start_date': start_date,
            'end_date': end_date,
            'gmvp_weights': dict(zip(tickers, gmvp_weights * 100)),
            'tp_weights': dict(zip(tickers, tp_weights * 100)),
            'gmvp_return': round(gmvp_exp * 100, 3),
            'gmvp_variance': round(gmvp_var, 3),
            'gmvp_std': round(gmvp_std * 100, 3),
            'gmvp_sharpe': round(gmvp_sharpe, 3),
            'tp_return': round(tp_exp * 100, 3),
            'tp_variance': round(tp_var, 3),
            'tp_std': round(tp_std * 100, 3),
            'tp_sharpe': round(tp_sharpe, 3),
            'cov_matrix_html': pd.DataFrame(cov_matrix, index=tickers, columns=tickers)
                                 .round(3).to_html(),
            'returns_html': pd.DataFrame(
                {'Expected Return (%)': expected_returns * 100},
                index=tickers
            ).round(3).to_html(),
            'frontier_data': frontier_data,
            'risk_free_rate': round(risk_free_rate * 100, 3),
            'gmvp_point': {'rischio': gmvp_std, 'rendimento': gmvp_exp},
            'tp_point': {'rischio': tp_std, 'rendimento': tp_exp},
            'gmvp_cumulative': gmvp_cumulative.to_json(date_format='iso'),
            'tp_cumulative': tp_cumulative.to_json(date_format='iso'),
        }

    return render(request, 'portfolios2/portfolios.html', context)
