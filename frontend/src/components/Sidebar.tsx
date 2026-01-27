/**
 * Sidebar Navigation Component
 */

import { NavLink } from 'react-router-dom';
import { FiFolder, FiMap, FiSettings, FiMessageSquare } from 'react-icons/fi';

export default function Sidebar() {
  const navItems = [
    { to: '/projects', icon: FiFolder, label: 'Projects' },
    { to: '/map', icon: FiMap, label: 'Map View' },
    { to: '/processing', icon: FiSettings, label: 'Processing' },
    { to: '/chat', icon: FiMessageSquare, label: 'GeoBot Assistant' },
  ];

  return (
    <aside className="w-64 bg-card border-r border-border flex flex-col">
      <div className="p-4 border-b border-border">
        <h1 className="text-2xl font-bold">🌍 GeoBot</h1>
      </div>
      
      <nav className="flex-1 p-4">
        <ul className="space-y-2">
          {navItems.map((item) => (
            <li key={item.to}>
              <NavLink
                to={item.to}
                className={({ isActive }) =>
                  `flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${
                    isActive
                      ? 'bg-primary text-primary-foreground'
                      : 'hover:bg-muted'
                  }`
                }
              >
                <item.icon className="w-5 h-5" />
                <span>{item.label}</span>
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
    </aside>
  );
}
