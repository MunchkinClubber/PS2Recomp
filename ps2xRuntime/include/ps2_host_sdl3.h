#pragma once

// SDL3 + Vulkan host backend (PS2X_HOST_SDL3).
//
// The runtime was written against raylib. This header provides the small part of the raylib API
// it uses (window, presenting one texture, keyboard/gamepad state, audio streams and one-shot
// sounds) with the same names and meanings, implemented in src/lib/host/ps2_host_sdl3.cpp on SDL3
// for the window/input/audio and Vulkan for presentation. No SDL or Vulkan headers leak from here.

#include <cstdint>

struct Color
{
    unsigned char r, g, b, a;
};

struct Vector2
{
    float x, y;
};

struct Rectangle
{
    float x, y, width, height;
};

struct Image
{
    void *data;
    int width;
    int height;
    int mipmaps;
    int format;
};

struct Texture2D
{
    unsigned int id;
    int width;
    int height;
    int mipmaps;
    int format;
};
using Texture = Texture2D;

struct AudioStream
{
    void *buffer; // host stream handle
    void *processor;
    unsigned int sampleRate;
    unsigned int sampleSize;
    unsigned int channels;
};

struct Wave
{
    unsigned int frameCount;
    unsigned int sampleRate;
    unsigned int sampleSize;
    unsigned int channels;
    void *data;
};

struct Sound
{
    AudioStream stream;
    unsigned int frameCount;
};

using AudioCallback = void (*)(void *bufferData, unsigned int frames);

#define LIGHTGRAY Color{200, 200, 200, 255}
#define GRAY Color{130, 130, 130, 255}
#define DARKGRAY Color{80, 80, 80, 255}
#define WHITE Color{255, 255, 255, 255}
#define BLACK Color{0, 0, 0, 255}
#define BLANK Color{0, 0, 0, 0}
#define MAGENTA Color{255, 0, 255, 255}
#define RED Color{230, 41, 55, 255}
#define GREEN Color{0, 228, 48, 255}
#define BLUE Color{0, 121, 241, 255}
#define YELLOW Color{253, 249, 0, 255}

enum ConfigFlags : unsigned int
{
    FLAG_VSYNC_HINT = 0x00000040,
    FLAG_FULLSCREEN_MODE = 0x00000002,
    FLAG_WINDOW_RESIZABLE = 0x00000004,
};

// Keyboard keys: the values are USB HID usage IDs (= SDL scancodes).
enum KeyboardKey
{
    KEY_NULL = 0,
    KEY_A = 4, KEY_B, KEY_C, KEY_D, KEY_E, KEY_F, KEY_G, KEY_H, KEY_I, KEY_J, KEY_K, KEY_L, KEY_M,
    KEY_N, KEY_O, KEY_P, KEY_Q, KEY_R, KEY_S, KEY_T, KEY_U, KEY_V, KEY_W, KEY_X, KEY_Y, KEY_Z,
    KEY_ONE = 30, KEY_TWO, KEY_THREE, KEY_FOUR, KEY_FIVE, KEY_SIX, KEY_SEVEN, KEY_EIGHT, KEY_NINE, KEY_ZERO,
    KEY_ENTER = 40,
    KEY_ESCAPE = 41,
    KEY_BACKSPACE = 42,
    KEY_TAB = 43,
    KEY_SPACE = 44,
    KEY_F1 = 58, KEY_F2, KEY_F3, KEY_F4, KEY_F5, KEY_F6, KEY_F7, KEY_F8, KEY_F9, KEY_F10, KEY_F11, KEY_F12,
    KEY_RIGHT = 79,
    KEY_LEFT = 80,
    KEY_DOWN = 81,
    KEY_UP = 82,
    KEY_KP_1 = 89, KEY_KP_2, KEY_KP_3, KEY_KP_4, KEY_KP_5, KEY_KP_6, KEY_KP_7, KEY_KP_8, KEY_KP_9,
    KEY_KP_0 = 98,
    KEY_LEFT_CONTROL = 224,
    KEY_LEFT_SHIFT = 225,
    KEY_LEFT_ALT = 226,
    KEY_RIGHT_CONTROL = 228,
    KEY_RIGHT_SHIFT = 229,
    KEY_RIGHT_ALT = 230,
};

// Same numbering as raylib.
enum GamepadButton
{
    GAMEPAD_BUTTON_UNKNOWN = 0,
    GAMEPAD_BUTTON_LEFT_FACE_UP,
    GAMEPAD_BUTTON_LEFT_FACE_RIGHT,
    GAMEPAD_BUTTON_LEFT_FACE_DOWN,
    GAMEPAD_BUTTON_LEFT_FACE_LEFT,
    GAMEPAD_BUTTON_RIGHT_FACE_UP,
    GAMEPAD_BUTTON_RIGHT_FACE_RIGHT,
    GAMEPAD_BUTTON_RIGHT_FACE_DOWN,
    GAMEPAD_BUTTON_RIGHT_FACE_LEFT,
    GAMEPAD_BUTTON_LEFT_TRIGGER_1,
    GAMEPAD_BUTTON_LEFT_TRIGGER_2,
    GAMEPAD_BUTTON_RIGHT_TRIGGER_1,
    GAMEPAD_BUTTON_RIGHT_TRIGGER_2,
    GAMEPAD_BUTTON_MIDDLE_LEFT,
    GAMEPAD_BUTTON_MIDDLE,
    GAMEPAD_BUTTON_MIDDLE_RIGHT,
    GAMEPAD_BUTTON_LEFT_THUMB,
    GAMEPAD_BUTTON_RIGHT_THUMB,
};

enum GamepadAxis
{
    GAMEPAD_AXIS_LEFT_X = 0,
    GAMEPAD_AXIS_LEFT_Y = 1,
    GAMEPAD_AXIS_RIGHT_X = 2,
    GAMEPAD_AXIS_RIGHT_Y = 3,
    GAMEPAD_AXIS_LEFT_TRIGGER = 4,
    GAMEPAD_AXIS_RIGHT_TRIGGER = 5,
};

enum TraceLogLevel
{
    LOG_ALL = 0, LOG_TRACE, LOG_DEBUG, LOG_INFO, LOG_WARNING, LOG_ERROR, LOG_FATAL, LOG_NONE
};

// Window / frame
void SetConfigFlags(unsigned int flags);
void InitWindow(int width, int height, const char *title);
void CloseWindow();
bool WindowShouldClose();
bool IsWindowReady();
void SetTargetFPS(int fps);
int GetScreenWidth();
int GetScreenHeight();
void SetTraceLogLevel(int logLevel);
void BeginDrawing();
void EndDrawing();
void ClearBackground(Color color);

// Images / textures (one texture drawn per frame is all the runtime needs)
Image GenImageColor(int width, int height, Color color);
void UnloadImage(Image image);
Texture2D LoadTextureFromImage(Image image);
void UpdateTexture(Texture2D texture, const void *pixels);
void UnloadTexture(Texture2D texture);
void DrawTexturePro(Texture2D texture, Rectangle source, Rectangle dest, Vector2 origin, float rotation, Color tint);

// Input (state is sampled once per frame on the main thread; safe to read from any thread)
bool IsKeyDown(int key);
bool IsKeyPressed(int key);
bool IsGamepadAvailable(int gamepad);
bool IsGamepadButtonDown(int gamepad, int button);
float GetGamepadAxisMovement(int gamepad, int axis);

// Audio
void InitAudioDevice();
void CloseAudioDevice();
bool IsAudioDeviceReady();
void SetAudioStreamBufferSizeDefault(int size);
AudioStream LoadAudioStream(unsigned int sampleRate, unsigned int sampleSize, unsigned int channels);
void UnloadAudioStream(AudioStream stream);
void SetAudioStreamCallback(AudioStream stream, AudioCallback callback);
void PlayAudioStream(AudioStream stream);
Wave LoadWaveFromMemory(const char *fileType, const unsigned char *fileData, int dataSize);
void UnloadWave(Wave wave);
Sound LoadSoundFromWave(Wave wave);
void UnloadSound(Sound sound);
void PlaySound(Sound sound);
void StopSound(Sound sound);
bool IsSoundPlaying(Sound sound);
void SetSoundPitch(Sound sound, float pitch);
void SetSoundVolume(Sound sound, float volume);
