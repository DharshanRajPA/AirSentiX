let sentimentChart;

function renderSentimentChart(scores) {
    const ctx = document.getElementById('sentimentChart').getContext('2d');
    const labels = Object.keys(scores);
    const data = Object.values(scores).map(score => score * 100);

    if (sentimentChart) sentimentChart.destroy();

    sentimentChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Confidence (%)',
                data,
                backgroundColor: ['#6366f1', '#10b981', '#f59e0b'],
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: { beginAtZero: true, max: 100 }
            }
        }
    });
}
