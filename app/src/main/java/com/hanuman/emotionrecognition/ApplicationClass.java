package com.hanuman.emotionrecognition;

import android.app.Application;

import com.onesignal.OneSignal;

public class ApplicationClass extends Application {

    @Override
    public void onCreate() {
        super.onCreate();
        // OneSignal Initialization
        OneSignal.startInit(this)
                .inFocusDisplaying(OneSignal.OSInFocusDisplayOption.Notification)
                .unsubscribeWhenNotificationsAreDisabled(true)
                .init();
    }
}
