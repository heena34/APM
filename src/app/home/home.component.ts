import { Component, OnInit } from '@angular/core';
import { LocationStrategy, PlatformLocation, Location } from '@angular/common';
import { LegendItem, ChartType } from '../lbd/lbd-chart/lbd-chart.component';
import * as Chartist from 'chartist';
import { Http } from '@angular/http';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})
export class HomeComponent implements OnInit {
  parameters: object;
  data: any;

  public emailChartType: ChartType;
  public emailChartData: any;
  public emailChartLegendItems: LegendItem[];

  public RULChartType: ChartType;
  public RULChartData: any = {};
  public RULChartOptions: any;
  public RULChartResponsive: any[];
  public RULChartLegendItems: LegendItem[];

  public activityChartType: ChartType;
  public activityChartData: any;
  public activityChartOptions: any;
  public activityChartResponsive: any[];
  public activityChartLegendItems: LegendItem[];
  graphFlag: Boolean = false;
  constructor(private http: Http) {
  }

  ngOnInit() {

    this.http.get('./assets/json/RUL.json').subscribe(res => {
      this.data = res.json();
      this.RULChartData = this.data['RUL'];
      this.activityChartData = this.data['PredictVSActual'];
      this.plotGraph();
      this.graphFlag = true;
    });
  }

  plotGraph() {
    this.emailChartType = ChartType.Pie;
    this.emailChartData = {
      labels: ['83%', '11%', '6%'],
      series: [62, 32, 6]
    };

    this.emailChartLegendItems = [
      { title: 'Cycle', imageClass: 'fa fa-circle text-info' },
      { title: 'RUL', imageClass: 'fa fa-circle text-danger' },
      { title: 'Total Life', imageClass: 'fa fa-circle text-warning' }
    ];


    this.RULChartType = ChartType.Bar;


    console.log('RULChartData', this.RULChartData)

    this.RULChartOptions = {
      low: 0,
      high: 500,
      showArea: true,
      height: '245px',
      axisX: {
        showGrid: false,
      },
      lineSmooth: Chartist.Interpolation.simple({
        divisor: 3
      }),
      showLine: false,
      showPoint: false,
    };
    this.RULChartResponsive = [
      ['', {
        axisX: {
          labelInterpolationFnc: function (value) {
            return value[0];
          }
        }
      }]
    ];
    this.RULChartLegendItems = [
      { title: 'RUL', imageClass: 'fa fa-circle text-info' },

    ];


    this.activityChartType = ChartType.Bar;

    this.activityChartOptions = {
      seriesBarDistance: 10,
      axisX: {
        showGrid: false
      },
      height: '245px'
    };
    this.activityChartResponsive = [
      ['', {
        seriesBarDistance: 5,
        axisX: {
          labelInterpolationFnc: function (value) {
            return value[0];
          }
        }
      }]
    ];
    this.activityChartLegendItems = [
      { title: 'Prediction', imageClass: 'fa fa-circle text-info' },
      { title: 'Actual', imageClass: 'fa fa-circle text-danger' }
    ];
  }

}
