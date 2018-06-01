import { Injectable } from '@angular/core';
import { Http, Response, ResponseContentType, RequestOptions } from '@angular/http';
import { HttpModule, Headers } from '@angular/http';

import { Observable } from 'rxjs/Observable';
import 'rxjs/add/operator/map';
import 'rxjs/add/operator/share';
import 'rxjs/add/operator/startWith';

import { concat } from 'rxjs/operator/concat';
import { TroposLoadingComponent } from 'app/tropos.tools/tropos.loading/tropos.loading.component';

@Injectable()
export class ApiService {
    [x: string]: any;
    baseUrl = '/';
    //baseUrl = 'http://localhost:80/';
    constructor(private http: Http) { }
    public Get(url): Observable<any> {
        return this.http.get(this.baseUrl + url);
    }
    public Post(url, params): Observable<any> {
        return this.http.post(this.baseUrl + url, params);

    }
    public PostFile(url, files: File[]): Observable<any> {
        const formData: FormData = new FormData();
        formData.append('file', files[0], files[0].name);
        return this.http.post(this.baseUrl + url, formData);
    }
    public showLoader() {
        TroposLoadingComponent.isLoading = true;

        setTimeout(() => this.hideLoader(),120000);
    }
    public hideLoader() {
        TroposLoadingComponent.isLoading = false;
    }

}
