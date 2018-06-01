import { Component, Input } from '@angular/core';


@Component({
    selector: 'app-tropos-loading',
    templateUrl: './tropos.loading.component.html',
    styleUrls: ['./tropos.loading.component.css']
})
export class TroposLoadingComponent {
    static isLoading: boolean;
    constructor() {
        TroposLoadingComponent.isLoading = false;
    }
    get showLoader() {
        return TroposLoadingComponent.isLoading;

    }


}
