## Completed


# Add app icon in flutter mobile app

To add an app icon to a Flutter mobile app, follow these steps:

1. Create a new folder in the root of your Flutter project called "assets". This is where you will store your app icons.
    
2. Add your app icons to the "assets" folder. Make sure to include versions of the icon for different screen densities, such as 1x, 2x, 3x, and 4x. You can generate these versions using a tool like [**https://makeappicon.com/**](https://makeappicon.com/).
    
3. Open the `pubspec.yaml` file in the root of your Flutter project. This file contains the configuration for your Flutter app, including the assets that it uses.
    
4. Under the "flutter" section, add an entry for each of your app icons to the "assets" list. For example:
    

```yaml
flutter_icons:
  android: "launcher_icon"
  ios: true
  image_path: "assets/icon.png"
```

1. Run the following command to update the Flutter app with the new icons:
    

```bash
flutter pub add flutter_launcher_icons
flutter pub get
flutter pub run flutter_launcher_icons:main
```

Now hot reload the app.
That's it! Your Flutter app should now use the app icon that you specified.

# Hope you like this... Happy coding...
