import { Component, ViewChild, ElementRef, Renderer2, OnInit, OnDestroy } from '@angular/core';
import { Router, NavigationExtras } from '@angular/router';
import { ApiService } from 'app/common/services/api.service';
import { LegendItem, ChartType } from '../lbd/lbd-chart/lbd-chart.component';
import { Http, Response, Headers, RequestOptions } from '@angular/http';
import * as Chartist from 'chartist';
import { Observable } from 'rxjs/Rx';
import 'rxjs/add/operator/map';
import 'rxjs/add/operator/catch';
import { TroposAlertService } from 'app/common/services/tropos.alert.service';


@Component({
  selector: 'app-user',
  templateUrl: './admin.component.html',
  styleUrls: ['./admin.component.css']
})
export class AdminComponent implements OnInit {
  imageName: String;
  imageType: String;
  imageSize: String;
  name: String;

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
  graphData = null;
  binarydata;
  jsonData;



  @ViewChild('fileInput') el: ElementRef;
  @ViewChild('fileInput1') el1: ElementRef;
  @ViewChild('fileInput2') el2: ElementRef;
  // tslint:disable-next-line:max-line-length
  constructor(private rd: Renderer2, private router: Router, private dataService: ApiService, private http: Http, private alertService: TroposAlertService) {
    this.name = 'upload file';

  }

  fileUpload() {

    if (this.el.nativeElement.files.length === 0 && this.el1.nativeElement.files.length === 0 && this.el2.nativeElement.files.length === 0) {
      this.alertService.error("Please select file before upload");
      return;
    }

    console.log("file size")
    console.log(this.el.nativeElement.files.length)
    console.log(this.el1.nativeElement.files.length)
    console.log(this.el2.nativeElement.files.length)


    if(this.el.nativeElement.files.length === 1) {
      this.dataService.PostFile('upload', this.el.nativeElement.files).subscribe(data => {
        try {
          this.alertService.success("Test data uploaded Successfully");
        } catch (e) {
          this.alertService.error("Uploading Test data failed");
        }
      });
    }
    else if(this.el1.nativeElement.files.length === 1)
    {
      this.dataService.PostFile('upload', this.el1.nativeElement.files).subscribe(data => {
        try {
          this.alertService.success("Train data uploaded Successfully");
        } catch (e) {
          this.alertService.error("Uploading Train data failed");
        }
      });
    }
    else if(this.el2.nativeElement.files.length === 1)
    {
      this.dataService.PostFile('upload', this.el2.nativeElement.files).subscribe(data => {
        try {
          this.alertService.success("Truth data uploaded Successfully");
        } catch (e) {
          this.alertService.error("Uploading Truth data failed");
        }
      });
    }
  }


  trainData() {
    //this.dataService.showLoader();
    this.alertService.success("trained");
    this.dataService.Get('train').subscribe(data => {

      if (data.status === 200) {
        //this.dataService.hideLoader();
      }
    });
  }



 testData() {
    console.log("test data");
    this.binarydata = {"w0":"0","w1":"15"};


     this.dataService.Post('test', this.binarydata).subscribe(data => {
      try {
        console.log(data);
        if (data.status === 200) {
          console.log("this.graphData::",data);
          this.graphData = data._body;
          console.log("gd::",this.graphData);
          console.log("engineID::",this.graphData['engine_id']);
          //this.jsonData = JSON.parse(this.graphData)
          //console.log("gd::",this.jsonData);
        }
      } catch (e) {
        this.alertService.error("Testing Data Fail");
      }
    }, error => {
      this.alertService.error("Testing Data Fail");
    });

  }

  ngOnInit() {

      this.plotGraph();
      this.graphFlag = true;

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
    //console.log(JSON.parse(this.graphData)['engine_id']);
    //this.RULChartData['labels'] = this.graphData;
    //this.RULChartData['series'] = this.graphData;
  
    // this.RULChartData['labels'] = this.graphData['engine_id'];
    // this.RULChartData['series'] = this.graphData['actual_rul'];
    this.RULChartData = {
      "labels": [
         1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100
      ],
      "series": [
          [
             112, 98, 69, 82, 91, 93, 91, 95, 111, 96, 97, 124, 95, 107, 83, 84, 50, 28, 87, 16, 57, 111, 113, 20, 145, 119, 66, 97, 90, 115, 8, 48, 106, 7, 11, 19, 21, 50, 142, 28, 18, 10, 59, 109, 114, 47, 135, 92, 21, 79, 114, 29, 26, 97, 137, 15, 103, 37, 114, 100, 21, 54, 72, 28, 128, 14, 77, 8, 121, 94, 118, 50, 131, 126, 113, 10, 34, 107, 63, 90, 8, 9, 137, 58, 118, 89, 116, 115, 136, 28, 38, 20, 85, 55, 128, 137, 82, 59, 117, 20
          ]
      ]
  };




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
    this.activityChartData = {
      labels: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16'],
      series: [
        [542, 443, 320, 780, 553, 453, 326, 434, 568, 610, 756, 895, 780, 553, 453, 326,],
        [412, 243, 280, 580, 453, 353, 300, 364, 368, 410, 636, 695, 280, 580, 453, 353]
      ]
    }

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
