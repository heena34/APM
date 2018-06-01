import { Injectable } from '@angular/core';
import { Router, NavigationStart } from '@angular/router';
import { Observable } from 'rxjs/Observable';
import { Subject } from 'rxjs/Subject';
import { Alert, AlertType, AlertBoxType, AlertBox } from '../classes/alert';



@Injectable()
export class TroposAlertService {

    private subject = new Subject<Alert>();
    private alertBoxSubject = new Subject<AlertBox>();
    private keepAfterRouteChange = false;

    constructor(private router: Router) {
        // clear alert messages on route change unless 'keepAfterRouteChange' flag is true
        router.events.subscribe(event => {
            if (event instanceof NavigationStart) {
                if (this.keepAfterRouteChange) {
                    // only keep for a single route change
                    this.keepAfterRouteChange = false;
                } else {
                    // clear alert messages
                    this.clear();
                }
            }
        });
    }

    getAlert(): Observable<any> {
        return this.subject.asObservable();
    }

    getAlertBox(): Observable<any> {
        return this.alertBoxSubject.asObservable();
    }

    // short alerts

    success(message: string, keepAfterRouteChange = true) {
        this.alert(AlertType.Success, message, keepAfterRouteChange);
    }

    error(message: string, keepAfterRouteChange = false) {
        this.alert(AlertType.Error, message, keepAfterRouteChange);
    }

    info(message: string, keepAfterRouteChange = false) {
        this.alert(AlertType.Info, message, keepAfterRouteChange);
    }

    warn(message: string, keepAfterRouteChange = false) {
        this.alert(AlertType.Warning, message, keepAfterRouteChange);
    }

    alert(type: AlertType, message: string, keepAfterRouteChange = false) {
        this.keepAfterRouteChange = keepAfterRouteChange;
        this.subject.next(<Alert>{ type: type, message: message });
    }

    // alert box
    alertBoxError(message: string, keepAfterRouteChange = false) {
        this.alertBox(AlertBoxType.Error, message, keepAfterRouteChange);
    }

    alertBoxInfo(message: string, keepAfterRouteChange = false) {
        this.alertBox(AlertBoxType.Info, message, keepAfterRouteChange);
    }

    alertBoxSuccess(message: string, keepAfterRouteChange = false) {
        this.alertBox(AlertBoxType.Success, message, keepAfterRouteChange);
    }

    alertBox(type: AlertBoxType, message: string, keepAfterRouteChange = false) {
        this.keepAfterRouteChange = keepAfterRouteChange;
        this.alertBoxSubject.next(<AlertBox>{ type: type, message: message });
    }

    clear() {
        // clear alerts
        this.subject.next();
        this.alertBoxSubject.next();
    }
}
