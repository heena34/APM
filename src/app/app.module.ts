import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HttpModule } from '@angular/http';
import { RouterModule } from '@angular/router';

import { AppRoutingModule } from './app.routing';
import { NavbarModule } from './shared/navbar/navbar.module';
import { FooterModule } from './shared/footer/footer.module';
import { SidebarModule } from './sidebar/sidebar.module';
import { LbdModule } from './lbd/lbd.module';

import { AppComponent } from './app.component';

import { HomeComponent } from './home/home.component';
import { AdminComponent } from './admin/admin.component';
import { TablesComponent } from './tables/tables.component';
import { TypographyComponent } from './typography/typography.component';
import { IconsComponent } from './icons/icons.component';
import { MapsComponent } from './maps/maps.component';
import { NotificationsComponent } from './notifications/notifications.component';
//import { FusionChartsModule } from 'angular2-fusioncharts';

// Import FusionCharts library
//import * as FusionCharts from 'fusioncharts';
// Import FusionCharts Charts module
//import * as Charts from 'fusioncharts/fusioncharts.charts';
import { ApiService } from 'app/common/services/api.service';

import { TroposLoadingComponent } from 'app/tropos.tools/tropos.loading/tropos.loading.component';
import { TroposAlertComponent } from 'app/tropos.tools/tropos.alert/tropos.alert.component';
import { TroposAlertService } from 'app/common/services/tropos.alert.service';
import { LiveStreamingComponent } from './live-streaming/live-streaming.component';

@NgModule({
  declarations: [
    AppComponent,
    HomeComponent,
    AdminComponent,
    TablesComponent,
    TypographyComponent,
    IconsComponent,
    MapsComponent,
    NotificationsComponent,
    TroposLoadingComponent,
    TroposAlertComponent,
    LiveStreamingComponent,

  ],
  imports: [
    BrowserModule,
    FormsModule,
    HttpModule,
    NavbarModule,
    FooterModule,
    SidebarModule,
    RouterModule,
    AppRoutingModule,
    LbdModule,
    //FusionChartsModule.forRoot(FusionCharts, Charts)
  ],
  providers: [ApiService, TroposAlertService],
  bootstrap: [AppComponent]
})
export class AppModule { }
