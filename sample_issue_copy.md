**1. App Crashes on Launch (iOS 18 Beta)**\
Since updating to iOS 18 beta, the app crashes immediately after launch.
I've tried reinstalling and clearing storage, but the crash persists.
Can you confirm compatibility with the new OS?

**2. Add Dark Mode Toggle**\
Currently, the app follows the system theme, but some users want to set
their own preference. Could we add a manual dark mode toggle in
Settings? This would help when testing visuals or when users prefer a
fixed theme.

**3. Clarify API Authentication Docs**\
The [API Authentication](https://example.com/docs/auth) page skips
explaining how to refresh tokens. For example:

``` bash
curl -X POST https://api.example.com/auth/refresh   -d "refresh_token=abc123"
```

It would help to show expected responses and error codes.

**4. Slow Load Time on Dashboard**\
The dashboard takes \~10 seconds to load on a stable WiFi connection.
Other pages load quickly, so maybe the issue is with how widgets or
charts are being initialized. Can this be optimized?

**5. Question: Bulk Import for CSV**\
Is there a way to import CSV files with more than 10,000 rows? I see
smaller files work fine, but large uploads stall without errors. If it's
unsupported, could we note the file size limits in the docs?

**6. Feature Request: Keyboard Shortcuts**\
It would be great to have keyboard shortcuts for common actions (e.g.,
`⌘+S` to save, `⌘+F` to search). This would improve productivity for
power users who rely on the app daily.

**7. Login Error with Special Characters**\
Users with special characters in their email addresses (e.g., `+`, `%`)
can't log in. Example:

    test+user@example.com

It works at signup but fails on login. Looks like an encoding issue.

**8. Documentation Example Uses Deprecated Method**\
The docs still mention `fetchDataLegacy()` in the [Data API
section](https://example.com/docs/data), but the SDK now uses
`fetchData()`. Updating this would prevent confusion for new developers.

**9. Mobile View Cuts Off Navigation**\
On smaller Android devices, the bottom navigation bar is partially
hidden behind the system gesture area. Users need to swipe multiple
times to reach icons. Can we add padding or safe area handling?

**10. Support Webhooks for Events**\
Right now, we only support polling the API for event updates. Could we
add webhook support (e.g., `POST` to a callback URL)? This would make
integrations more real-time and reduce unnecessary requests.
