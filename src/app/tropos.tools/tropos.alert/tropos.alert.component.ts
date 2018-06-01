
import { Component, OnInit } from '@angular/core';
import { Alert, AlertType, AlertBox, AlertBoxType } from '../../common/classes/alert';
import { TroposAlertService } from '../../common/services/tropos.alert.service';

@Component({
    selector: 'app-tropos-alert',
    templateUrl: './tropos.alert.component.html',
    styleUrls: ['./tropos.alert.component.css']
})

export class TroposAlertComponent implements OnInit {
    alerts: Alert[] = [];
    alertboxes: AlertBox[] = [];

    constructor(private alertService: TroposAlertService) { }

    ngOnInit() {
        this.alertService.getAlert().subscribe((alert: Alert) => {
            if (!alert) {
                // clear alerts when an empty alert is received
                this.alerts = [];
                return;
            }

            // add alert to array
            this.alerts.push(alert);

            // remove alert after 5 seconds
            setTimeout(() => this.removeAlert(alert), 5000);
        });

        this.alertService.getAlertBox().subscribe((alertbox: AlertBox) => {
            if (!alertbox) {
                // clear alerts when an empty alert is received
                this.alertboxes = [];
                return;
            }
            // add alert to array
            this.alertboxes.push(alertbox);
        });
    }

    removeAlert(alert: Alert) {
        this.alerts = this.alerts.filter(x => x !== alert);
    }

    removeAlertBox(alertbox: AlertBox) {
        this.alertboxes = this.alertboxes.filter(x => x !== alertbox);
    }

    cssClass(alert: Alert) {
        if (!alert) {
            return;
        }

        // return css class based on alert type
        switch (alert.type) {
            case AlertType.Success:
                return 'alert-message alert-success-message';
            case AlertType.Error:
                return 'alert-message alert-danger-message';
            case AlertType.Info:
                return 'alert-message alert-info-message';
            case AlertType.Warning:
                return 'alert-message alert-warning-message';
        }
    }

    alertBoxCssClass(alertBox: AlertBox) {
        if (!alertBox) {
            return;
        }

        // return css class based on confirmationBox type
        switch (alertBox.type) {
            case AlertBoxType.Error:
                return 'alertbox alert-danger-message';
            case AlertBoxType.Info:
                return 'alertbox alert-info-message';
            case AlertBoxType.Success:
                return 'alertbox alert-warning-message';
        }
    }

}
