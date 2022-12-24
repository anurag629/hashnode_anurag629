# Add app icon in flutter mobile app

To add an app icon to a Flutter mobile app, follow these steps:

1. Create a new folder in the root of your Flutter project called "assets". This is where you will store your app icons.
    
2. Add your app icons to the "assets" folder. Make sure to include versions of the icon for different screen densities, such as 1x, 2x, 3x, and 4x. You can generate these versions using a tool like [**https://makeappicon.com/**](https://makeappicon.com/).
    
3. Open the `pubspec.yaml` file in the root of your Flutter project. This file contains the configuration for your Flutter app, including the assets that it uses.
    
4. Under the "flutter" section, add an entry for each of your app icons to the "assets" list. For example:
    

```yaml
flutter:
  assets:
    - assets/icon/icon.png
    - assets/icon/icon_2x.png
    - assets/icon/icon_3x.png
    - assets/icon/icon_4x.png
```

1. Run the following command to update the Flutter app with the new icons:
    

```bash
flutter pub get
```

1. Finally, specify the app icon in the `AndroidManifest.xml` file for Android and the `Info.plist` file for iOS. For example, in the `AndroidManifest.xml` file, you would add the following line:
    

```xml
<application
    android:icon="@mipmap/ic_launcher"
    ...>
```

Replace "ic\_launcher" with the name of your app icon file.

That's it! Your Flutter app should now use the app icon that you specified.

# Hope you like this... Happy coding...