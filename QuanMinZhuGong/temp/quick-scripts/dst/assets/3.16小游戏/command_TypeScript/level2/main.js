
                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level2/main.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, '91522+UAMROjo6f/8WxasmZ', 'main');
// 3.16小游戏/command_TypeScript/level2/main.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var main_1 = /** @class */ (function (_super) {
    __extends(main_1, _super);
    function main_1() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    main_1_1 = main_1;
    main_1.prototype.start = function () {
        this.node.on('touchstart', this.onEventStart, this);
    };
    main_1.prototype.onEventStart = function () {
        main_1_1.attack = true;
    };
    main_1.prototype.update = function (dt) {
        cc.log("min1, min2, min3 " + main_1_1.minion1_hp + " " + main_1_1.minion3_hp + " " + main_1_1.minion3_hp);
    };
    var main_1_1;
    // LIFE-CYCLE CALLBACKS:
    // onLoad () {}
    main_1.attack = false;
    main_1.minion_attack = false;
    main_1.main_hp = 163;
    main_1.minion1_hp = 163;
    main_1.minion2_hp = 163;
    main_1.minion3_hp = 163;
    __decorate([
        property(cc.Label)
    ], main_1.prototype, "label", void 0);
    __decorate([
        property
    ], main_1.prototype, "text", void 0);
    main_1 = main_1_1 = __decorate([
        ccclass
    ], main_1);
    return main_1;
}(cc.Component));
exports.default = main_1;

cc._RF.pop();
                    }
                    if (nodeEnv) {
                        __define(__module.exports, __require, __module);
                    }
                    else {
                        __quick_compile_project__.registerModuleFunc(__filename, function () {
                            __define(__module.exports, __require, __module);
                        });
                    }
                })();
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDJcXG1haW4udHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7OztBQUFBLG9CQUFvQjtBQUNwQix3RUFBd0U7QUFDeEUsbUJBQW1CO0FBQ25CLGtGQUFrRjtBQUNsRiw4QkFBOEI7QUFDOUIsa0ZBQWtGOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFFNUUsSUFBQSxLQUFzQixFQUFFLENBQUMsVUFBVSxFQUFsQyxPQUFPLGFBQUEsRUFBRSxRQUFRLGNBQWlCLENBQUM7QUFHMUM7SUFBb0MsMEJBQVk7SUFBaEQ7UUFBQSxxRUE4QkM7UUEzQkcsV0FBSyxHQUFhLElBQUksQ0FBQztRQUd2QixVQUFJLEdBQVcsT0FBTyxDQUFDOztJQXdCM0IsQ0FBQztlQTlCb0IsTUFBTTtJQW9CdkIsc0JBQUssR0FBTDtRQUNJLElBQUksQ0FBQyxJQUFJLENBQUMsRUFBRSxDQUFDLFlBQVksRUFBRSxJQUFJLENBQUMsWUFBWSxFQUFFLElBQUksQ0FBQyxDQUFDO0lBQ3hELENBQUM7SUFDRCw2QkFBWSxHQUFaO1FBQ0ksUUFBTSxDQUFDLE1BQU0sR0FBRyxJQUFJLENBQUM7SUFFekIsQ0FBQztJQUNELHVCQUFNLEdBQU4sVUFBUSxFQUFFO1FBQ04sRUFBRSxDQUFDLEdBQUcsQ0FBQyxtQkFBbUIsR0FBQyxRQUFNLENBQUMsVUFBVSxHQUFDLEdBQUcsR0FBQyxRQUFNLENBQUMsVUFBVSxHQUFDLEdBQUcsR0FBQyxRQUFNLENBQUMsVUFBVSxDQUFDLENBQUE7SUFDN0YsQ0FBQzs7SUFyQkQsd0JBQXdCO0lBRXhCLGVBQWU7SUFFRixhQUFNLEdBQVksS0FBSyxDQUFDO0lBQ3hCLG9CQUFhLEdBQVksS0FBSyxDQUFDO0lBQy9CLGNBQU8sR0FBVyxHQUFHLENBQUM7SUFDdEIsaUJBQVUsR0FBVyxHQUFHLENBQUM7SUFDekIsaUJBQVUsR0FBVyxHQUFHLENBQUM7SUFDekIsaUJBQVUsR0FBVyxHQUFHLENBQUM7SUFkdEM7UUFEQyxRQUFRLENBQUMsRUFBRSxDQUFDLEtBQUssQ0FBQzt5Q0FDSTtJQUd2QjtRQURDLFFBQVE7d0NBQ2M7SUFOTixNQUFNO1FBRDFCLE9BQU87T0FDYSxNQUFNLENBOEIxQjtJQUFELGFBQUM7Q0E5QkQsQUE4QkMsQ0E5Qm1DLEVBQUUsQ0FBQyxTQUFTLEdBOEIvQztrQkE5Qm9CLE1BQU0iLCJmaWxlIjoiIiwic291cmNlUm9vdCI6Ii8iLCJzb3VyY2VzQ29udGVudCI6WyIvLyBMZWFybiBUeXBlU2NyaXB0OlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy90eXBlc2NyaXB0Lmh0bWxcclxuLy8gTGVhcm4gQXR0cmlidXRlOlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy9yZWZlcmVuY2UvYXR0cmlidXRlcy5odG1sXHJcbi8vIExlYXJuIGxpZmUtY3ljbGUgY2FsbGJhY2tzOlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy9saWZlLWN5Y2xlLWNhbGxiYWNrcy5odG1sXHJcblxyXG5jb25zdCB7Y2NjbGFzcywgcHJvcGVydHl9ID0gY2MuX2RlY29yYXRvcjtcclxuXHJcbkBjY2NsYXNzXHJcbmV4cG9ydCBkZWZhdWx0IGNsYXNzIG1haW5fMSBleHRlbmRzIGNjLkNvbXBvbmVudCB7XHJcblxyXG4gICAgQHByb3BlcnR5KGNjLkxhYmVsKVxyXG4gICAgbGFiZWw6IGNjLkxhYmVsID0gbnVsbDtcclxuXHJcbiAgICBAcHJvcGVydHlcclxuICAgIHRleHQ6IHN0cmluZyA9ICdoZWxsbyc7XHJcblxyXG4gICAgLy8gTElGRS1DWUNMRSBDQUxMQkFDS1M6XHJcblxyXG4gICAgLy8gb25Mb2FkICgpIHt9XHJcbiAgIFxyXG4gICBwdWJsaWMgc3RhdGljIGF0dGFjazogYm9vbGVhbiA9IGZhbHNlO1xyXG4gICBwdWJsaWMgc3RhdGljIG1pbmlvbl9hdHRhY2s6IGJvb2xlYW4gPSBmYWxzZTtcclxuICAgcHVibGljIHN0YXRpYyBtYWluX2hwOiBudW1iZXIgPSAxNjM7XHJcbiAgIHB1YmxpYyBzdGF0aWMgbWluaW9uMV9ocDogbnVtYmVyID0gMTYzO1xyXG4gICBwdWJsaWMgc3RhdGljIG1pbmlvbjJfaHA6IG51bWJlciA9IDE2MztcclxuICAgcHVibGljIHN0YXRpYyBtaW5pb24zX2hwOiBudW1iZXIgPSAxNjM7XHJcbiAgIFxyXG5cclxuICAgIHN0YXJ0ICgpIHtcclxuICAgICAgICB0aGlzLm5vZGUub24oJ3RvdWNoc3RhcnQnLCB0aGlzLm9uRXZlbnRTdGFydCwgdGhpcyk7XHJcbiAgICB9XHJcbiAgICBvbkV2ZW50U3RhcnQoKSB7XHJcbiAgICAgICAgbWFpbl8xLmF0dGFjayA9IHRydWU7XHJcbiAgICAgICBcclxuICAgIH1cclxuICAgIHVwZGF0ZSAoZHQpIHtcclxuICAgICAgICBjYy5sb2coXCJtaW4xLCBtaW4yLCBtaW4zIFwiK21haW5fMS5taW5pb24xX2hwK1wiIFwiK21haW5fMS5taW5pb24zX2hwK1wiIFwiK21haW5fMS5taW5pb24zX2hwKVxyXG4gICAgfVxyXG59XHJcbiJdfQ==