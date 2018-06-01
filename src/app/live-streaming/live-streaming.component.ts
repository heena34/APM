import { Component, OnInit } from '@angular/core';
import { Http } from '@angular/http';
import { LegendItem, ChartType } from '../lbd/lbd-chart/lbd-chart.component';
import * as Chartist from 'chartist';
import { Observable } from 'rxjs/Rx';
import 'rxjs/add/operator/map';
import 'rxjs/add/operator/catch';
import { LocationStrategy, PlatformLocation, Location } from '@angular/common';


@Component({
  selector: 'app-live-streaming',
  templateUrl: './live-streaming.component.html',
  styleUrls: ['./live-streaming.component.scss']
})
export class LiveStreamingComponent implements OnInit {
  public RULChartType: ChartType;
  public RULChartData: any = {};
  public RULChartOptions: any;
  public RULChartResponsive: any[];
  public RULChartLegendItems: LegendItem[];
  data: any;
  plotFlag: Boolean = false;
  timer;
  newValue;

  constructor(private http: Http) { }

  ngOnInit() {

    // this.http.get('./assets/json/dummy.json').subscribe(res => {
    //   this.data = res.json();
    //   this.RULChartData = {
    //     labels: this.data['time'],
    //     series: [this.data['Predicted']]
    //   };
    //   //console.log(this.RULChartData)
    //   this.plotGraph();
    //   this.plotFlag = true;
    //   //console.log(this.data);
    // });

    this.timer = Observable.timer(0, 10000);

    this.newValue = () => {
      return this.timer
        .flatMap((i) => this.http.get('./assets/json/dummy.json'))
    }

    console.log(this.newValue);
    this.RULChartData = {
      labels: this.data['time'],
      series: [this.data['Predicted']]
    };
    this.plotGraph();
    this.plotFlag = true;
  }


  plotGraph() {

    this.RULChartType = ChartType.Bar;
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
      { title: 'RUL life', imageClass: 'fa fa-circle text-info' },

    ];
  }


}
