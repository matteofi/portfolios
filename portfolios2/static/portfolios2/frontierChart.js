document.addEventListener('DOMContentLoaded', function () {
    const frontierElement = document.getElementById('frontier-data');
    const gmvpElement = document.getElementById('gmvp-point');
    const tpElement = document.getElementById('tp-point');
    if (!frontierElement || !gmvpElement || !tpElement) return;

    const frontierData = JSON.parse(frontierElement.textContent);
    const gmvp = JSON.parse(gmvpElement.textContent);
    const tp = JSON.parse(tpElement.textContent);

    const rischi = frontierData.map(p => p.rischio);
    const rendimenti = frontierData.map(p => p.rendimento);
    const sharpe = frontierData.map(p => p.sharpe);

    const frontierTrace = {
        x: rischi,
        y: rendimenti,
        mode: 'markers',
        type: 'scatter',
        name: 'Frontiera Efficiente',
        marker: {
            size: 6,
            color: sharpe,
            colorscale: [
                [0.0, 'rgb(0, 0, 255)'],
                [0.1, 'rgb(28, 28, 230)'],
                [0.2, 'rgb(56, 56, 204)'],
                [0.3, 'rgb(85, 85, 179)'],
                [0.4, 'rgb(113, 113, 153)'],
                [0.5, 'rgb(141, 56, 128)'],
                [0.6, 'rgb(170, 28, 102)'],
                [0.7, 'rgb(198, 0, 77)'],
                [0.8, 'rgb(226, 0, 51)'],
                [1.0, 'rgb(255, 0, 25)']
            ],
            colorbar: {
                title: 'Sharpe Ratio',
            }
        }
    };

    const gmvpTrace = {
        x: [gmvp.rischio],
        y: [gmvp.rendimento],
        mode: 'markers',
        type: 'scatter',
        name: 'GMVP',
        marker: {
            color: 'green',
            size: 12,
            symbol: 'star'
        }
    };

    const tpTrace = {
        x: [tp.rischio],
        y: [tp.rendimento],
        mode: 'markers',
        type: 'scatter',
        name: 'TP',
        marker: {
            color: 'green',
            size: 12,
            symbol: 'diamond'
        }
    };

    const layout = {
        title: 'Efficient frontier',
        xaxis: { title: 'Risk (Standard Deviation)' },
        yaxis: { title: 'Expected return' },
        margin: { t: 40, r: 80, b: 60, l: 60 },
        legend: {
            orientation: 'h',
            y: -0.2
        }
    };

    Plotly.newPlot('grafico', [frontierTrace, gmvpTrace, tpTrace], layout);
});



const gmvpCumulativeElement = document.getElementById('gmvp-cumulative');
const tpCumulativeElement = document.getElementById('tp-cumulative');

if (gmvpCumulativeElement && tpCumulativeElement) {
    const gmvpCumulative = JSON.parse(gmvpCumulativeElement.textContent);
    const tpCumulative = JSON.parse(tpCumulativeElement.textContent);

    const gmvpDates = Object.keys(gmvpCumulative);
    const gmvpValues = Object.values(gmvpCumulative);

    const tpDates = Object.keys(tpCumulative);
    const tpValues = Object.values(tpCumulative);

    const gmvpTraceLine = {
        x: gmvpDates,
        y: gmvpValues,
        type: 'scatter',
        mode: 'lines',
        name: 'GMVP',
        line: { color: 'blue' }
    };

    const tpTraceLine = {
        x: tpDates,
        y: tpValues,
        type: 'scatter',
        mode: 'lines',
        name: 'TP',
        line: { color: 'red' }
    };

    const layoutLine = {
        title: 'Andamento del Portafoglio nel Tempo',
        xaxis: { title: 'Data' },
        yaxis: { title: 'Cumulative Return' },
        margin: { t: 40, r: 80, b: 60, l: 60 },
        legend: { orientation: 'h', y: -0.2 }
    };

    const chartDiv = document.createElement('div');
    chartDiv.id = 'portfolio-performance';
    chartDiv.style = 'width:100%; max-width:800px; height:500px; margin:auto; margin-top:30px;';
    document.body.appendChild(chartDiv);

    Plotly.newPlot('portfolio-performance', [gmvpTraceLine, tpTraceLine], layoutLine);
}
