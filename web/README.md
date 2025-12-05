# 💬 Constitution GPT - Chat Interface

A beautiful, modern chat interface for Constitution GPT built with Next.js 16, React 19, and Tailwind CSS v4.

## ✨ Features

### 💬 Chat Features
- **Message Bubbles**: Distinct styling for user (gradient) and AI (glass) messages
- **Suggested Questions**: 6 pre-defined constitutional questions for quick start
- **Auto-Resize Textarea**: Input field grows with content (max 200px)
- **Loading States**: Elegant 3-dot pulse animation while waiting for responses
- **Timestamps**: Each message shows the time it was sent
- **Keyboard Shortcuts**: 
  - `Enter` to send message
  - `Shift + Enter` for new line
- **Clear Input**: Quick clear button when typing

## � Installation & Setup

### Prerequisites
- Node.js 20+ installed
- npm or yarn package manager

### Installation Steps

1. **Install dependencies** (if not already installed):
   ```bash
   npm install
   ```

2. **Start the development server**:
   ```bash
   npm run dev
   ```

3. **Open your browser**:
   Navigate to [http://localhost:3000](http://localhost:3000)

The chat interface should now be running! 🎉

### Build for Production

To create a production build:

```bash
npm run build
npm start
```

## 📖 How to Use

### Getting Started

1. **Open the application** at `http://localhost:3000`
2. **Welcome Screen**: You'll see a welcome message with the Constitution GPT logo
3. **Suggested Questions**: Click on any of the 6 suggested questions to get started quickly

### Asking Questions

**Method 1: Use Suggested Questions**
- Click on any suggested question card
- The question will be sent automatically
- Wait for the AI response (currently simulated)

**Method 2: Type Your Own Question**
- Click on the input field at the bottom
- Type your question about the Constitution of Nepal
- Press `Enter` to send (or click the Send button)
- Use `Shift + Enter` to add a new line without sending

### Understanding the Interface

**Header Section**:
- 🏛️ Constitution GPT logo and title
- 🟢 Online status indicator

**Messages Area**:
- **User messages**: Purple gradient bubbles on the right
- **AI responses**: Glass-morphic bubbles on the left with Constitution GPT branding
- **Loading state**: Animated dots when waiting for response
- **Auto-scroll**: Automatically scrolls to the latest message

**Input Area**:
- Auto-resizing textarea (grows as you type)
- Clear button (X) appears when typing
- Send button (disabled when empty)
- Keyboard shortcuts hint at the bottom

### Current Functionality

**Note**: The chat interface is currently using **simulated responses** for demonstration purposes. The responses show the format and structure of how constitutional information will be displayed once connected to the RAG backend.

Example simulated response format:
```
📘 Part 7 – Federal Executive
Article 76 – Constitution of Council of Ministers

🔹 Sub-article (1)
As per Part 7, Article 76, Sub-article (1):
• The President shall appoint the leader of a parliamentary party...
```

### Modifying Animations

All animations are defined in `app/globals.css` using `@keyframes`:
- `fadeIn` - Fade in animation for messages
- `slideUp` - Slide up animation for suggested questions
- `pulse` - Pulsing animation for loading states
- `shimmer` - Shimmer effect (available for future use)

## 🐛 Troubleshooting

### Port Already in Use

If port 3000 is already in use:
```bash
# Use a different port
PORT=3001 npm run dev
```

### Hydration Errors

The chat interface uses `ssr: false` to prevent hydration errors. If you encounter any:
1. Clear your browser cache
2. Restart the development server
3. Check that `'use client';` is at the top of `page.tsx`

### Lint Warning: Unknown at rule @theme

This is a known warning with Tailwind CSS v4. It doesn't affect functionality and can be safely ignored. The `@theme` directive is a valid Tailwind v4 feature.

### Dark Mode Not Working

Dark mode is based on system preferences. To test:
- **macOS**: System Preferences → General → Appearance
- **Windows**: Settings → Personalization → Colors
- **Linux**: Depends on your desktop environment

## 🚀 Next Steps

Once you connect the backend:
1. Replace the simulated response in `ChatInterface.tsx` with actual API calls
2. Add error handling for failed requests
3. Implement message persistence (localStorage or database)
4. Add user authentication if needed
5. Add export functionality for conversations
6. Consider adding voice input for accessibility

## 📝 Technologies Used

- **Next.js 16** - React framework with App Router
- **React 19** - UI library
- **TypeScript** - Type safety
- **Tailwind CSS v4** - Utility-first CSS framework
- **CSS Custom Properties** - Theme customization
- **CSS Animations** - Smooth transitions and effects

## � License

MIT License - Same as the parent Constitution GPT project

---

Built with ❤️ for Constitution GPT
