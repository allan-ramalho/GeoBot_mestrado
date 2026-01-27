/**
 * Loading Screen Component
 */

export default function LoadingScreen() {
  return (
    <div className="flex h-screen items-center justify-center bg-background">
      <div className="text-center">
        <div className="mb-4 h-16 w-16 animate-spin rounded-full border-4 border-primary border-t-transparent mx-auto"></div>
        <h2 className="text-2xl font-bold mb-2">GeoBot</h2>
        <p className="text-muted-foreground">Initializing...</p>
      </div>
    </div>
  );
}
