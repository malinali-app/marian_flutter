# One-time Android Rust targets for Cargokit / marian_flutter.
# Requires rustup + Android NDK (via Android Studio).

$ErrorActionPreference = "Stop"

rustup target add `
  aarch64-linux-android `
  armv7-linux-androideabi `
  i686-linux-android `
  x86_64-linux-android

Write-Host "Android Rust targets installed."
Write-Host "Open the example app and run: flutter run"
Write-Host "Cargokit will compile rust/ into jniLibs during the Gradle build."
