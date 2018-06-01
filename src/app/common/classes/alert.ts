export class Alert {
    type: AlertType;
    message: string;
}

export enum AlertType {
    Success,
    Error,
    Info,
    Warning
}

export class AlertBox {
    type: AlertBoxType;
    message: string;
}

export enum AlertBoxType {
    Success,
    Error,
    Info
}
