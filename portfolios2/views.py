import yfinance as yf
import pandas as pd
import numpy as np
import cvxpy as cp
from datetime import datetime
from scipy.optimize import minimize
from django.shortcuts import render


# Funzioni gmvp, tangency_portfolio, efficient_frontier (NON modificate)


def portfolio_view(request):
    context = {}
    if request.method == 'POST':
        start_date = request.POST.get('start_date')
        end_date = request.POST.get('end_date')

        # Controllo date
        try:
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()

            if end_date_obj < start_date_obj:
                context['error'] = 'Errore: La data di fine deve essere successiva alla data di inizio.'
                return render(request, 'portfolios2/portfolios.html', context)
            if start_date_obj < datetime(1960, 1, 1).date():
                context['error'] = 'Errore: Data meno recente fissata a 01/01/1960.'
                return render(request, 'portfolios2/portfolios.html', context)
        except (ValueError, TypeError):
            context['error'] = 'Errore: Date fornite non valide.'
            return render(request, 'portfolios2/portfolios.html', context)

        tickers_input = request.POST.get('tickers', '')
        tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

        if len(tickers) < 2:
            context['error'] = 'Errore: Inserire almeno 2 ticker validi, separati da virgola.'
            return render(request, 'portfolios2/portfolios.html', context)

        short_selling = 'short_selling' in request.POST
        risk_free_choice = request.POST.get('risk_free_choice')
        custom_risk_free = request.POST.get('custom_risk_free')

        # Gestione tasso risk-free
        if risk_free_choice == '0':
            risk_free_rate = 0.0
        elif risk_free_choice == 'custom':
            try:
                risk_free_rate = float(custom_risk_free)
            except (ValueError, TypeError):
                context['error'] = 'Errore: Tasso risk-free personalizzato non valido (inserire valore numerico con punto).'
                return render(request, 'portfolios2/portfolios.html', context)
        elif risk_free_choice == 'irx':
            try:
                irx_data = yf.download("^IRX", start=start_date, end=end_date)['Close'].dropna()
                if irx_data.empty:
                    raise ValueError
                risk_free_rate = irx_data.mean() / 100
            except Exception:
                context['error'] = 'Errore nel recupero del tasso IRX.'
                return render(request, 'portfolios2/portfolios.html', context)
        else:
            risk_free_rate = 0.0

        try:
            raw_data = yf.download(tickers, start=start_date, end=end_date)
            data = raw_data.get('Close')
        
            if data is None or data.empty:
                raise ValueError('Dati di chiusura mancanti o vuoti')
        
            found_tickers = data.columns.levels[1] if isinstance(data.columns, pd.MultiIndex) else data.columns
            missing_tickers = [t for t in tickers if t not in found_tickers]
        
            if missing_tickers or data.empty:
                context['error'] = 'Errore in tickers.'
                return render(request, 'portfolios2/portfolios.html', context)
        
            data.dropna(inplace=True)
            if data.empty:
                context['error'] = 'Errore in tickers.'
                return render(request, 'portfolios2/portfolios.html', context)
        
        except Exception:
            context['error'] = 'Errore in tickers.'
            return render(request, 'portfolios2/portfolios.html', context)


        # Calcoli principali
        log_returns = np.log(data / data.shift(1)).dropna()
        cov_matrix = log_returns.cov() * 252
        expected_returns = log_returns.mean() * 252

        gmvp_weights = gmvp(cov_matrix, short_selling)
        gmvp_exp = expected_returns @ gmvp_weights
        gmvp_var = gmvp_weights.T @ cov_matrix @ gmvp_weights
        gmvp_std = np.sqrt(gmvp_var)

        tp_weights = tangency_portfolio(expected_returns, cov_matrix, short_selling=short_selling, risk_free_rate=risk_free_rate)
        tp_exp = expected_returns @ tp_weights
        tp_var = tp_weights.T @ cov_matrix @ tp_weights
        tp_std = np.sqrt(tp_var)

        frontier_data = efficient_frontier(cov_matrix, expected_returns, risk_free_rate=risk_free_rate)

        gmvp_sharpe = gmvp_exp / gmvp_std
        tp_sharpe = (tp_exp - risk_free_rate) / tp_std

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
            'cov_matrix_html': pd.DataFrame(cov_matrix, index=tickers, columns=tickers).round(3).to_html(),
            'returns_html': pd.DataFrame({'Ritorno atteso (%)': expected_returns * 100}, index=tickers).round(3).to_html(),
            'frontier_data': frontier_data,
            'risk_free_rate': round(risk_free_rate * 100, 3),
            'gmvp_point': {'rischio': gmvp_std, 'rendimento': gmvp_exp},
            'tp_point': {'rischio': tp_std, 'rendimento': tp_exp},
            'gmvp_cumulative': gmvp_cumulative.to_json(date_format='iso'),
            'tp_cumulative': tp_cumulative.to_json(date_format='iso'),
        }

    return render(request, 'portfolios2/portfolios.html', context)
