var ctx = document.getElementById('month_chart');
var month_chart = new Chart(ctx, {
    type: 'bar',
    data: {
        labels:
            ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        datasets: [{
            label: '14 ~ 19년도 월별 평균 한강이용객 수',
            data: [3395011, 3318623, 4910634, 7896458, 7422706, 6699639, 6763286, 8385902, 7291181, 7554729, 3468799, 2880115],
            backgroundColor: [
                'rgba(255, 99, 132, 0.35)',
                'rgba(54, 162, 235, 0.35)',
                'rgba(255, 206, 86, 0.35)',
                'rgba(75, 192, 192, 0.35)',
                'rgba(153, 102, 255, 0.35)',
                'rgba(255, 159, 64, 0.35)',
                'rgba(255, 99, 132, 0.35)',
                'rgba(54, 162, 235, 0.35)',
                'rgba(255, 206, 86, 0.35)',
                'rgba(75, 192, 192, 0.35)',
                'rgba(153, 102, 255, 0.35)',
                'rgba(255, 159, 64, 0.35)'
            ],
            borderColor: [
                'rgba(255, 99, 132, 1)',
                'rgba(54, 162, 235, 1)',
                'rgba(255, 206, 86, 1)',
                'rgba(75, 192, 192, 1)',
                'rgba(153, 102, 255, 1)',
                'rgba(255, 159, 64, 1)',
                'rgba(255, 99, 132, 1)',
                'rgba(54, 162, 235, 1)',
                'rgba(255, 206, 86, 1)',
                'rgba(75, 192, 192, 1)',
                'rgba(153, 102, 255, 1)',
                'rgba(255, 159, 64, 1)'
            ],
            borderWidth: 1
        }]
    },
    options: {
        scales: {
            yAxes: [{

                ticks: {
                    fontColor: 'white',
                    beginAtZero: true
                }
            }],
            xAxes: [{

                ticks: {
                    fontColor: 'white',
                    beginAtZero: true
                }
            }]
        }
    }
}); 